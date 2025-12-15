from typing import Literal

import pytest
import torch
from torch import nn
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma

# PaliGemma：处理视觉和语言输入的VLM
# Gemma专家模型：专门用来动作预测

class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,                 # VLM 配置
        action_expert_config,       # 动作专家配置
        use_adarms=None,            # 是否使用 AdaRMS 自适应归一化，默认None
        precision: Literal["bfloat16", "float32"] = "bfloat16",     # 模型精度
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        # 创建 PaliGemma 模型配置
        vlm_config_hf = CONFIG_MAPPING["paligemma"]()       # 创建对象  
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001  # 词汇表大小
        vlm_config_hf.image_token_index = 257152            # 图像token索引
        vlm_config_hf.text_config.hidden_size = vlm_config.width    # 隐藏层大小
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim    # 中间层大小
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads    # 注意力头数量
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim    # 注意力头维度
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth  # 隐藏层数量
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads # KV头数量
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"   # 激活函数
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152   # 词汇表大小
        vlm_config_hf.text_config.use_adarms = use_adarms[0]    # 是否使用 AdARSM
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None # AdaRSM条件维度
        vlm_config_hf.vision_config.intermediate_size = 4304    # 视觉中间层大小
        vlm_config_hf.vision_config.projection_dim = 2048       # 投影维度
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"  # 投影激活函数
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim, # 头维度
            hidden_size=action_expert_config.width, # 隐藏层大小
            intermediate_size=action_expert_config.mlp_dim, # 中间层大小
            num_attention_heads=action_expert_config.num_heads, # 头数量
            num_hidden_layers=action_expert_config.depth,   # 隐藏层数量
            num_key_value_heads=action_expert_config.num_kv_heads,  # KV头数量
            vocab_size=257152,  # 词汇表大小
            hidden_activation="gelu_pytorch_tanh",  # 激活函数
            torch_dtype="float32",
            use_adarms=use_adarms[1],   # 是否使用 AdARSM
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,   # AdaRSM条件维度
        )
        
        # 初始化模型组件
        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None # 禁用专家模型的词嵌入，使用PaliGemma的嵌入

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        # 设置模型参数为指定精度
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        # 保持FP32的参数列表
        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",      # 视觉模型的图像块嵌入权重
            "vision_tower.vision_model.embeddings.patch_embedding.bias",        # 视觉模型的图像块嵌入偏置
            "vision_tower.vision_model.embeddings.position_embedding.weight",   # 视觉模型的位置嵌入权重
            "input_layernorm",              # 输入层归一化
            "post_attention_layernorm",     # 注意力后的层归一化
            "model.norm",   # 模型归一化
        ]

        # 遍历参数，转化回fp32
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        # 图像嵌入
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        # 语言嵌入
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,     # 注意力掩码
        position_ids: torch.LongTensor | None = None,   # 位置id
        past_key_values: list[torch.FloatTensor] | pytest.Cache | None = None,  # 过去的KV
        inputs_embeds: list[torch.FloatTensor] | None = None,   # 输入嵌入
        use_cache: bool | None = None,      # 是否使用缓存
        adarms_cond: list[torch.Tensor] | None = None,  # AdaRMS 条件
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        
        # 只有 PaiGemma 输入（VLM）
        if inputs_embeds[1] is None:
            # 使用 PaliGemma 语言模型处理输出
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0], # 语言嵌入
                attention_mask=attention_mask,  # 注意力掩码
                position_ids=position_ids,      # 位置ID
                past_key_values=past_key_values,    # 过去的KV
                use_cache=use_cache,    # 是否使用缓存
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,    # AdaRMS 条件
            )
            prefix_past_key_values = prefix_output.past_key_values  # 获取过去的 KV
            prefix_output = prefix_output.last_hidden_state  # PaliGemma 语言模型最后的隐藏状态
            suffix_output = None    # Gemma 专家输出为None
        # 只有 Gemma 专家输入
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state # Gemma 专家最后的隐藏状态
            prefix_output = None    # PaliGemma的输入为None
            prefix_past_key_values = None # PaliGemma没有 KV
        # 两种都有，执行联合处理
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]   # 模型列表
            num_layers = self.paligemma.config.text_config.num_hidden_layers    # 层数

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Force enable gradient checkpointing if we're in training mode and the model supports it
            if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                if not self.gemma_expert.model.gradient_checkpointing:
                    print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                    self.gemma_expert.model.gradient_checkpointing = True
                use_gradient_checkpointing = True

            # Debug gradient checkpointing status
            if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}")
                print(f"Model training mode: {self.training}")
                print(
                    f"Gemma expert model has gradient_checkpointing attr: {hasattr(self.gemma_expert.model, 'gradient_checkpointing')}"
                )
                if hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                    print(
                        f"Gemma expert model gradient_checkpointing value: {self.gemma_expert.model.gradient_checkpointing}"
                    )
                self._debug_gc_printed = True

            # Define the complete layer computation function for gradient checkpointing
            # 单层的完整前向传播
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                models = [self.paligemma.language_model, self.gemma_expert.model]

                query_states = []   # Q 列表
                key_states = []     # K 列表
                value_states = []   # V 列表
                gates = []  # 门控列表
                # 对每个模型的输入进行处理
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]  # 输入形状
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim) # 隐藏层形状
                    # 计算 Q，K，V，加入到list
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Concatenate and process attention
                # 连接注意力
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                # 创建虚拟张量用于旋转位置嵌入
                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                # 计算旋转位置嵌入的正余弦并应用
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling # 缩放因子

                # Attention computation
                # 计算注意力
                att_output, _ = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,  # 自注意力层
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling,
                )
                # Get head_dim from the current layer, not from the model
                # 当前层的头维度
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)   # 改变注意力层输出形状

                # Process layer outputs
                # 处理层输出
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    # 确保数据类型匹配
                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    # 输出投影
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # first residual
                    # 第一个残差连接
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()  # 第一个残差后的状态
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    # 前馈网络
                    out_emb = layer.mlp(out_emb)
                    # second residual
                    # 第二个残差连接
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos # 跟新起始位置

                return outputs_embeds

            # Process all layers with gradient checkpointing if enabled
            # 使用checkpoint处理所有层
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    # 直接计算
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

                # Old code removed - now using compute_layer_complete function above

            # final norm
            # Define final norm computation function for gradient checkpointing
            # 最终归一化函数
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            # 获取输出
            prefix_output = outputs_embeds[0]   # PaliGemma 输出
            suffix_output = outputs_embeds[1]   # Gemma 专家输出
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values

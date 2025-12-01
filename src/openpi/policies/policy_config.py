import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    # 从训练好的 checkpoint 创建一个策略（Policy），可直接用于推理或评估
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    # 如果没有提供 repack_transforms，就用空的 transforms.Group()
    repack_transforms = repack_transforms or transforms.Group()
    # checkpoint 自动下载（远程 URL 或本地路径都可用）
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # model.safetensors 文件存在 → PyTorch 模型
    # 否则 → 可能是 JAX / Flax 模型
    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        # 调用模型类的 load_pytorch 方法加载权重
        model = train_config.model.load_pytorch(train_config, weight_path)
        # 对部分参数转换为 bfloat16，减少显存占用，提高推理速度
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        # 调用 load + _model.restore_params 从 checkpoint 恢复权重
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    
    # 根据训练配置创建数据管线配置
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    # 如果没有提供 norm_stats，就从 checkpoint 加载
    # 归一化统计用于数据预处理（标准化、归一化,保证推理时和训练一致
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # 自动选择 GPU 或 CPU;如果环境没有 PyTorch，默认使用 CPU
    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    # 返回 _policy.Policy 对象，可以直接调用 policy.sample_actions() 生成动作
    return _policy.Policy(
        ''' 
        model → 训练好的模型
        transforms → 模型输入前的变换列表：
            可选的 repack_transforms
            注入默认语言 prompt
            数据管线预处理
            归一化
            模型特定输入变换
        output_transforms → 模型输出后的变换列表：
            模型特定输出变换
            去归一化(Unnormalize)
            数据管线输出变换
            repack_transforms 输出
        sample_kwargs → 调用 sample_actions() 时的参数
        metadata → 策略元信息
        is_pytorch / pytorch_device → PyTorch 模型运行设备
        '''
        
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )

import math
import re

import flax.linen as nn
import flax.struct as struct
import jax.numpy as jnp

import openpi.shared.array_typing as at


@struct.dataclass
class LoRAConfig:
    """Configuration for LoRA."""
    # LoRA 配置类

    # LoRA rank.
    rank: int
    # LoRA scaling factor.
    # LoRA 缩放系数，控制 LoRA 增量的强度
    alpha: float = 1.0
    
    # Initialization function for LoRA parameters.
    # LoRA 参数初始化函数（默认使用正态分布初始化）
    init_fn: nn.initializers.Initializer = nn.initializers.normal(stddev=0.01)

    # Enable rank-stabilized LoRA: https://arxiv.org/pdf/2312.03732
    # 是否启用 RSLoRA (rank-stabilized LoRA),在不同 rank 下保持增量大小稳定
    rslora: bool = False

    # Axes in the weight to apply LoRA to. Should typically be the last two axes.
    # 在权重张量上应用 LoRA 的两个轴，通常为倒数第二和倒是第一轴
    axes: tuple[int, int] = (-2, -1)
    
    # Axis label which is used by LoRA in einsum equations. Must not be present in the original equation.
    # 在 einsum 方程中使用的额外标签字符，要求原方程不包含该标签,确保不冲突
    label: str = "L"

    @property
    def scaling_value(self) -> float:
        # 返回 LoRA 增量的缩放值
        # 启用 RSLoRA 缩放为 alpha / sqrt(rank) ，否则为 alpha / rank
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank


class Einsum(nn.Module):
    """Einsum with LoRA support. Can be used as a drop-in replacement for the Gemma Einsum."""
    # 带 LoRA 的 Einsum 模块：对 jnp.einsum 进行包装，支持在权重上应用 LoRA
    # 提供 lora_config 时，构造两段式 LoRA 等式 （A，B）进行低秩近似
    
    # Shape of the weight.
    # 权重形状，用于确定参数维度
    shape: tuple[int, ...]
    # Initialization function for the weight.
    # 权重初始化，默认全0
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    # If not None, apply LoRA to the weight.
    lora_config: LoRAConfig | None = None

    def setup(self):
        # 模型初始化

        # 注册主权重参数 w
        self.w = self.param("w", self.init_fn, self.shape)

        # 若提供了 LoRA 配置，设置 LoRA 的两个分解权重 AB
        if config := self.lora_config:
            # Setup LoRA parameters.
            # 根据主权重形状复制两个列表，用于修改
            shape_a, shape_b = list(self.shape), list(self.shape)
            shape_a[config.axes[1]] = config.rank
            shape_b[config.axes[0]] = config.rank
            # 注册 LoRA 的 A，B参数
            self.w_a = self.param("lora_a", config.init_fn, shape_a)
            self.w_b = self.param("lora_b", config.init_fn, shape_b)

    @nn.compact
    def __call__(self, eqn: str, x):
        # 前向计算
        # eqn：einsum 的字符串表达式，例如 abc，cde -> abe

        # 保存输入张量的原始 dtype
        dtype = x.dtype  # original dtype, could be half-precision
        # 基准路径：直接对 x 和 w 调用 einsum 的结果
        result = jnp.einsum(eqn, x, self.w.astype(dtype))

        # 如果使用 LoRA，计算 LoRA
        if config := self.lora_config:
            eqn_a, eqn_b = self._make_lora_eqns(eqn)
            lora = jnp.einsum(eqn_a, x, self.w_a.astype(dtype))
            lora = jnp.einsum(eqn_b, lora, self.w_b.astype(dtype))
            result = result + lora * config.scaling_value

        return result

    def _make_lora_eqns(self, eqn: str) -> tuple[str, str]:
        if "L" in eqn:
            raise ValueError(f"L already in eqn: {eqn}")
        if not (m := re.match("(.*),(.*)->(.*)", eqn)):
            raise ValueError(f"Unsupported einsum eqn: {eqn}")
        lhs, rhs, out = m.groups()

        assert self.lora_config is not None
        a_label, b_label = (rhs[x] for x in self.lora_config.axes)
        label = self.lora_config.label

        a_rhs = rhs.replace(b_label, label)
        a_out = out.replace(b_label, label)
        eqn_a = f"{lhs},{a_rhs}->{a_out}"

        b_rhs = rhs.replace(a_label, label)
        eqn_b = f"{a_out},{b_rhs}->{out}"

        return eqn_a, eqn_b


class FeedForward(nn.Module):
    """Feed forward module."""

    features: int
    hidden_dim: int
    # If not None, apply LoRA to the weight.
    lora_config: LoRAConfig | None = None

    def setup(self):
        self.w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        )
        self.w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        )
        self.w_gating_lora = None
        self.w_linear_lora = None
        if self.lora_config:
            # Setup LoRA parameters.
            # TODO: follow up with a simplified init_fn api.
            self.w_gating_lora = (
                self.param("gating_einsum_lora_a", self.lora_config.init_fn, (2, self.features, self.lora_config.rank)),
                self.param(
                    "gating_einsum_lora_b", self.lora_config.init_fn, (2, self.lora_config.rank, self.hidden_dim)
                ),
            )
            self.w_linear_lora = (
                self.param("linear_lora_a", self.lora_config.init_fn, (self.hidden_dim, self.lora_config.rank)),
                self.param("linear_lora_b", self.lora_config.init_fn, (self.lora_config.rank, self.features)),
            )

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # original dtype, could be half-precision
        ff_gate = self._dot(
            x,
            self.w_gating[0],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][0], self.w_gating_lora[1][0]),
        )
        gate_value = nn.gelu(ff_gate)

        ff1 = self._dot(
            x,
            self.w_gating[1],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][1], self.w_gating_lora[1][1]),
        )
        activations = gate_value * ff1

        outputs = self._dot(activations, self.w_linear, self.w_linear_lora)
        assert outputs.dtype == dtype
        return outputs

    def _dot(self, x: at.Array, w: at.Array, lora_weights: tuple[at.Array, at.Array] | None) -> at.Array:
        base = jnp.dot(x, w.astype(x.dtype))
        if lora_weights is None:
            return base
        return base + jnp.dot(jnp.dot(x, lora_weights[0].astype(x.dtype)), lora_weights[1].astype(x.dtype))

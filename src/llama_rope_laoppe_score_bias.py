import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


class DistanceGatedLAOPPEBias(nn.Module):
    """
    Distance-Gated Residual LA-OPPE Bias.

    核心设计：
    1. 不修改 Q/K 主路径。
    2. RoPE attention score 完全保留。
    3. LA-OPPE 只作为 attention score residual bias。
    4. 使用相对距离 |i-j|，而不是绝对位置 m。
    5. alpha 可学习，初始化为 0，因此初始模型严格等价 baseline。
    """

    def __init__(
        self,
        num_heads: int,
        threshold: int = 2048,
        temperature: float = 512.0,
        max_len: int = 131072,
        alpha_max: float = 0.05,
        init_std: float = 0.02,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.threshold = threshold
        self.temperature = temperature
        self.max_len = max_len
        self.alpha_max = alpha_max

        # 每个 head 一组 LA-OPPE 特征权重。
        # 特征包括：
        # 1. log(distance)
        # 2. sin(log(distance))
        # 3. cos(log(distance))
        # 4. constant bias
        self.feature_weight = nn.Parameter(
            torch.empty(num_heads, 4)
        )
        nn.init.normal_(self.feature_weight, mean=0.0, std=init_std)

        # 每个 head 一个可学习 alpha。
        # 初始化为 0，使得初始 residual bias = 0。
        self.alpha_raw = nn.Parameter(
            torch.zeros(num_heads)
        )

        # 用于日志记录
        self.last_alpha_abs_mean = 0.0
        self.last_gate_short = 0.0
        self.last_gate_long = 0.0
        self.last_bias_abs_mean = 0.0

    def forward(self, q_len: int, kv_len: int, device, dtype):
        """
        返回 shape:
            [1, num_heads, q_len, kv_len]
        可直接加到 attention score 上。

        关键优化：
        1. 尽量全程用目标 dtype（bf16）计算，避免 float32 巨型中间张量。
        2. 用 clamp_ 原地操作，避免再次分配 2GB 级别张量。
        """

        work_dtype = dtype

        q_start = kv_len - q_len

        q_pos = torch.arange(
            q_start,
            q_start + q_len,
            device=device,
            dtype=work_dtype,
        )

        k_pos = torch.arange(
            0,
            kv_len,
            device=device,
            dtype=work_dtype,
        )

        # 相对距离 |i-j|: [q_len, kv_len]
        distance = torch.abs(q_pos[:, None] - k_pos[None, :])

        long_gate = torch.sigmoid(
            (distance - self.threshold) / self.temperature
        )

        log_d = torch.log1p(distance) / math.log1p(float(self.max_len))
        sin_log = torch.sin(math.pi * log_d)
        cos_log = torch.cos(math.pi * log_d)

        # feature_weight 转成目标 dtype，避免 float32 扩大中间张量
        w = self.feature_weight.to(dtype=work_dtype)

        raw_bias = (
            w[:, 0, None, None] * log_d[None, :, :]
            + w[:, 1, None, None] * sin_log[None, :, :]
            + w[:, 2, None, None] * cos_log[None, :, :]
            + w[:, 3, None, None]
        )

        alpha = self.alpha_max * torch.tanh(
            self.alpha_raw.to(dtype=work_dtype)
        )

        bias = alpha[:, None, None] * long_gate[None, :, :] * raw_bias

        # 原地 clamp，避免额外申请巨大张量
        bias.clamp_(min=-2.0, max=2.0)

        with torch.no_grad():
            effective_gate = torch.abs(alpha[:, None, None] * long_gate[None, :, :])

            short_mask = distance <= self.threshold
            long_mask = distance > self.threshold

            self.last_alpha_abs_mean = torch.abs(alpha).mean().item()

            if short_mask.any():
                self.last_gate_short = effective_gate[:, short_mask].mean().item()
            else:
                self.last_gate_short = 0.0

            if long_mask.any():
                self.last_gate_long = effective_gate[:, long_mask].mean().item()
            else:
                self.last_gate_long = 0.0

            self.last_bias_abs_mean = torch.abs(bias).mean().item()

        return bias.unsqueeze(0)


class LlamaAttentionLAOPPEScoreBias(nn.Module):
    """
    保留原始 LLaMA RoPE attention，
    只在 attention score 上添加 Distance-Gated LA-OPPE residual bias。
    """

    def __init__(
        self,
        old_attn,
        layer_idx: int,
        threshold: int = 2048,
        temperature: float = 512.0,
        max_len: int = 131072,
        alpha_max: float = 0.05,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.config = old_attn.config

        self.q_proj = old_attn.q_proj
        self.k_proj = old_attn.k_proj
        self.v_proj = old_attn.v_proj
        self.o_proj = old_attn.o_proj

        if hasattr(old_attn, "rotary_emb"):
            self.rotary_emb = old_attn.rotary_emb

        # 兼容不同 transformers 版本
        cfg = getattr(old_attn, "config", None)
        if cfg is None:
            raise AttributeError("old_attn 没有 config，无法推断 attention 维度。")

        self.hidden_size = getattr(
            old_attn,
            "hidden_size",
            getattr(cfg, "hidden_size", None),
        )
        if self.hidden_size is None:
            self.hidden_size = old_attn.q_proj.in_features

        self.num_heads = getattr(
            old_attn,
            "num_heads",
            getattr(cfg, "num_attention_heads", None),
        )
        if self.num_heads is None:
            raise AttributeError("无法从 old_attn/config 中推断 num_heads。")

        self.num_key_value_heads = getattr(
            old_attn,
            "num_key_value_heads",
            getattr(cfg, "num_key_value_heads", self.num_heads),
        )

        self.head_dim = getattr(old_attn, "head_dim", None)
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads

        self.num_key_value_groups = getattr(
            old_attn,
            "num_key_value_groups",
            self.num_heads // self.num_key_value_heads,
        )

        self.attention_dropout = getattr(
            old_attn,
            "attention_dropout",
            0.0,
        )

        self.is_causal = True

        self.laoppe_bias = DistanceGatedLAOPPEBias(
            num_heads=self.num_heads,
            threshold=threshold,
            temperature=temperature,
            max_len=max_len,
            alpha_max=alpha_max,
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz,
            q_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        key_states = key_states.view(
            bsz,
            q_len,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        value_states = value_states.view(
            bsz,
            q_len,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        # ------------------------------------------------------------
        # 原生 RoPE 主路径：完全保留
        # ------------------------------------------------------------
        if position_embeddings is not None:
            cos, sin = position_embeddings
        else:
            if not hasattr(self, "rotary_emb"):
                raise RuntimeError(
                    "当前 attention 没有 rotary_emb，且没有传入 position_embeddings。"
                )
            cos, sin = self.rotary_emb(value_states, position_ids)

        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
        )

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }

            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        key_states = repeat_kv(
            key_states,
            self.num_key_value_groups,
        )

        value_states = repeat_kv(
            value_states,
            self.num_key_value_groups,
        )

        kv_len = key_states.shape[-2]

                # ------------------------------------------------------------
        # 使用 fused SDPA，避免显式构造完整 softmax attention matrix
        # ------------------------------------------------------------
        residual_bias = self.laoppe_bias(
            q_len=q_len,
            kv_len=kv_len,
            device=query_states.device,
            dtype=query_states.dtype,
        )  # [1, H, Q, K]

        attn_mask_for_sdpa = residual_bias

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :kv_len].to(query_states.dtype)
            attn_mask_for_sdpa = attn_mask_for_sdpa + causal_mask

        dropout_p = self.attention_dropout if self.training else 0.0

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask_for_sdpa,
            dropout_p=dropout_p,
            is_causal=False,
        )

        attn_weights = None

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(
            bsz,
            q_len,
            self.hidden_size,
        )

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


def replace_llama_attention_score_bias(
    model,
    start_layer=None,
    end_layer=None,
    threshold: int = 2048,
    temperature: float = 512.0,
    max_len: int = 131072,
    alpha_max: float = 0.05,
):
    """
    接入 Distance-Gated Residual LA-OPPE Score Bias。

    推荐初始设置：
        start_layer=14, end_layer=16
    即只接入最后 2 层。

    如果最后 2 层稳定，再试：
        start_layer=12, end_layer=16
    即最后 4 层。
    """

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("没有找到 model.model.layers，当前模型可能不是 LLaMA 结构。")

    layers = model.model.layers
    num_layers = len(layers)

    if start_layer is None:
        start_layer = 14 if num_layers >= 16 else max(0, num_layers - 2)

    if end_layer is None:
        end_layer = num_layers

    print("=" * 80)
    print("启用 Distance-Gated Residual LA-OPPE Score Bias")
    print("=" * 80)
    print(f"总层数: {num_layers}")
    print(f"接入层范围: [{start_layer}, {end_layer})")
    print(f"threshold   = {threshold}")
    print(f"temperature = {temperature}")
    print(f"alpha_max   = {alpha_max}")
    print("=" * 80)

    replaced = 0

    for layer_idx, layer in enumerate(layers):
        if layer_idx < start_layer or layer_idx >= end_layer:
            continue

        old_attn = layer.self_attn

        try:
            old_param = next(old_attn.parameters())
            old_device = old_param.device
            old_dtype = old_param.dtype
        except StopIteration:
            old_device = torch.device("cpu")
            old_dtype = torch.float32

        new_attn = LlamaAttentionLAOPPEScoreBias(
            old_attn=old_attn,
            layer_idx=layer_idx,
            threshold=threshold,
            temperature=temperature,
            max_len=max_len,
            alpha_max=alpha_max,
        )

        new_attn.to(device=old_device, dtype=old_dtype)

        layer.self_attn = new_attn

        replaced += 1
        print(f"✅ LA-OPPE Score Bias 接入层: model.layers.{layer_idx}.self_attn")

    print("=" * 80)
    print(f"总接入层数: {replaced}")
    print("=" * 80)

    return model


def collect_laoppe_metrics(model):
    """
    收集所有 LA-OPPE 层的 gate / alpha 日志。
    """

    alpha_vals = []
    gate_short_vals = []
    gate_long_vals = []
    bias_vals = []

    for module in model.modules():
        if isinstance(module, DistanceGatedLAOPPEBias):
            alpha_vals.append(module.last_alpha_abs_mean)
            gate_short_vals.append(module.last_gate_short)
            gate_long_vals.append(module.last_gate_long)
            bias_vals.append(module.last_bias_abs_mean)

    if len(alpha_vals) == 0:
        return {}

    return {
        "laoppe_alpha_abs_mean": sum(alpha_vals) / len(alpha_vals),
        "laoppe_gate_short": sum(gate_short_vals) / len(gate_short_vals),
        "laoppe_gate_long": sum(gate_long_vals) / len(gate_long_vals),
        "laoppe_bias_abs_mean": sum(bias_vals) / len(bias_vals),
    }

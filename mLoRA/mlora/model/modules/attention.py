from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from mlora.model.args import LinearInfo, LLMModelArgs, ModelData
from mlora.model.modules import AdapterModel
from mlora.profiler import nvtx_range, set_backward_tracepoint

from .linear import Linear
from .rms_norm import RMSNorm


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # see the above ref
    left_part = x[..., : x.shape[-1] // 2]
    right_part = x[..., x.shape[-1] // 2 :]
    return torch.cat((-right_part, left_part), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # data shape is: batch_size * n_head * seq_len * n_dim
    xq_embed = (xq * cos) + (rotate_half(xq) * sin)
    xk_embed = (xk * cos) + (rotate_half(xk) * sin)
    return (xq_embed, xk_embed)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, n_kv_heads, seq_len, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, seq_len, head_dim)
    x = x.reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)
    return x


def precompute_rope_angle(
    dim: int, seq_len: int, theta: float, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    # this implement is different with facebooksearch/llama
    #   ref: https://github.com/huggingface/transformers/issues/25199
    angles = 1.0 / (theta ** (torch.arange(0, dim, 2).float().to(device) / dim))
    seq = torch.arange(seq_len, device=device, dtype=angles.dtype)
    emb = torch.outer(seq, angles)
    emb = torch.cat((emb, emb), dim=-1)

    emb.requires_grad_(False)
    # cos(angle), sin(angle)
    return (emb.cos(), emb.sin())


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if attention_mask is None:
        # Flash Attention with hardware-level causal masking — O(n) memory
        output = F.scaled_dot_product_attention(
            query, key, value, is_causal=True
        )
    else:
        # Legacy path: explicit additive mask (for padded or non-causal sequences)
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask
        )
    return output.transpose(1, 2).contiguous()


class Attention(torch.nn.Module):
    wq_: Linear
    wk_: Linear
    wv_: Linear
    wo_: Linear

    def __init__(self, layer_id: int, args: LLMModelArgs):
        super().__init__()

        # use layer id to local the adapter
        self.layer_id_: int = layer_id

        self.n_heads_ = args.n_heads_
        self.n_kv_heads_ = args.n_kv_heads_
        self.head_dim_ = args.dim_ // args.n_heads_
        self.n_rep_ = self.n_heads_ // self.n_kv_heads_

        # optional QK-Norm (used by Qwen3, not by Llama)
        self.q_norm_: Optional[RMSNorm] = None
        self.k_norm_: Optional[RMSNorm] = None

        # rope angle cos and sin
        self.cos_, self.sin_ = precompute_rope_angle(
            args.dim_ // args.n_heads_,
            args.max_seq_len_,
            args.rope_theta_,
            args.device_,
        )

    def forward(self, data: torch.Tensor, mask: torch.Tensor, input_args: ModelData):
        batch_size, max_seq_len, _ = data.shape

        xq = self.wq_.forward(data, input_args)
        xk = self.wk_.forward(data, input_args)
        xv = self.wv_.forward(data, input_args)

        # conver shape to multi head
        # the shape is batch_size * number_of_head * seq_len * dim_of_head
        xq = xq.view(batch_size, max_seq_len, self.n_heads_, self.head_dim_).transpose(
            1, 2
        )
        xk = xk.view(
            batch_size, max_seq_len, self.n_kv_heads_, self.head_dim_
        ).transpose(1, 2)
        xv = xv.view(
            batch_size, max_seq_len, self.n_kv_heads_, self.head_dim_
        ).transpose(1, 2)

        # QK-Norm (Qwen3): normalize Q and K before RoPE
        if self.q_norm_ is not None:
            xq = self.q_norm_.forward(xq)
            xk = self.k_norm_.forward(xk)

        # apply rotary embedding (use cache_position_ offset for correct absolute positions)
        assert xq.dtype == xk.dtype
        pos = getattr(input_args, 'cache_position_', 0)
        cos = self.cos_[pos:pos + max_seq_len].to(xq.dtype)
        sin = self.sin_[pos:pos + max_seq_len].to(xq.dtype)

        with nvtx_range("f_rotray_emb"):
            xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        set_backward_tracepoint(xq.grad_fn, "b_q_rope")
        set_backward_tracepoint(xk.grad_fn, "b_k_rope")

        # KV cache: concat cached K/V (stored before repeat_kv to save memory)
        kv_cache = getattr(input_args, 'kv_cache_', None)
        if kv_cache is not None:
            cached = kv_cache[self.layer_id_]
            if cached is not None:
                cached_k, cached_v = cached
                xk = torch.cat([cached_k, xk], dim=2)
                xv = torch.cat([cached_v, xv], dim=2)
            kv_cache[self.layer_id_] = (xk, xv)

        # for llama2 need to repeat the heads
        # before dim: batch_size, n_kv_head, seq_len, head_dim
        # after dim: batch_size, n_head, seq_len, head_dim
        xk = repeat_kv(xk, self.n_rep_)
        xv = repeat_kv(xv, self.n_rep_)

        set_backward_tracepoint(xk.grad_fn, "b_k_rep")
        set_backward_tracepoint(xv.grad_fn, "b_v_rep")

        # Detect sentinel: 1-element tensor means "use flash causal, no mask"
        effective_mask = None if mask.numel() == 1 else mask

        with nvtx_range("f_attention"):
            attention_score = scaled_dot_product_attention(xq, xk, xv, effective_mask)
        attention_score = attention_score.view(batch_size, max_seq_len, -1)
        set_backward_tracepoint(attention_score.grad_fn, "b_attention")

        # get output attention score
        return self.wo_.forward(attention_score, input_args)

    def from_pretrained(self, attn_layer: torch.nn.Module, norm_eps: float = 1e-6):
        self.wq_ = Linear(attn_layer.q_proj)
        self.wk_ = Linear(attn_layer.k_proj)
        self.wv_ = Linear(attn_layer.v_proj)
        self.wo_ = Linear(attn_layer.o_proj)

        # QK-Norm (Qwen3): RMSNorm on Q/K per-head before RoPE
        if hasattr(attn_layer, 'q_norm'):
            self.q_norm_ = RMSNorm(attn_layer.q_norm.weight, norm_eps)
            self.k_norm_ = RMSNorm(attn_layer.k_norm.weight, norm_eps)

    @property
    def linear_dict(self) -> Dict[str, Linear]:
        return {
            f"layers.{self.layer_id_}.self_attn.q_proj": self.wq_,
            f"layers.{self.layer_id_}.self_attn.k_proj": self.wk_,
            f"layers.{self.layer_id_}.self_attn.v_proj": self.wv_,
            f"layers.{self.layer_id_}.self_attn.o_proj": self.wo_,
        }

    def load_adapter(self, adapter_model: AdapterModel):
        for name, module in self.linear_dict.items():
            if name not in adapter_model:
                continue
            module.load_adapter(adapter_model[name])

    def offload_adapter(self, adapter_name: str):
        for _, module in self.linear_dict.items():
            module.offload_adapter(adapter_name)

    def linears_info(self) -> OrderedDict[str, LinearInfo]:
        ret_val = OrderedDict()

        for name, module in self.linear_dict.items():
            assert isinstance(module, Linear)
            ret_val[name] = LinearInfo(
                name_=name,
                in_dim_=module.weight_.in_features,
                out_dim_=module.weight_.out_features,
                base_weight_=module.weight_,
            )

        return ret_val

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Literal

from einops import rearrange
from torch.utils.checkpoint import checkpoint
import math
from dataclasses import dataclass

world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal['bf16', 'fp8'] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"

@dataclass
class ModelArgs:
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.


class Config:
    def __init__(self):
        self.vocab_size = 30522
        self.d_model = 4096
        self.n_layers = 2
        self.n_heads = 8
        self.d_kv_comp = 128
        self.d_rope = 16
        self.n_experts = 32
        self.n_shared = 2
        self.top_k = 2
        self.seq_len = 256
        self.batch_size = 1
        self.ffn_dim = 384
        self.device_groups = 4
        

config = Config()

class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.ffn_dim)
        self.w2 = nn.Linear(config.ffn_dim, config.d_model)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))

class RotaryEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



class MemoryOptimizedMLA(nn.Module):
    def  __init__(self):
        super().__init__()
        self.d_head = config.d_model // config.n_heads
        self.split_size = self.d_head - config.d_rope

        # projectins
        self.W_dkv = nn.Linear(config.d_model, config.d_kv_comp)
        self.W_dq = nn.Linear(config.d_model, config.d_kv_comp)

        self.W_uq = nn.Linear(config.d_kv_comp, config.n_heads * self.split_size)
        self.W_uk = nn.Linear(config.d_kv_comp, config.n_heads * self.split_size)
        self.W_uv = nn.Linear(config.d_kv_comp, config.n_heads * self.d_head)

        self.W_qr = nn.Linear(config.d_kv_comp, config.n_heads * config.d_rope)
        self.W_kr = nn.Linear(config.d_kv_comp, config.n_heads * config.d_rope)

        self.rotary = RotaryEmbedding(config.d_rope)
        self.output = nn.Linear(config.n_heads * config.d_head, config.d_model)

    def forward(self, x, mask=None):
        b, s, _ = x.size()

        # KV compression
        c_kv = self.W_dkv(x)
        k = self.W_uk(c_kv).view(b, s, config.n_heads, self.split_size)
        v = self.W_uv(c_kv).view(b, s, config.n_heads, self.split_size)

        # Query Compression
        c_q = self.W_dq(x)
        q_base  = self.W_qr(c_q).view(b,s,config.n_heads, self.split_size)
        q_rot  = self.W_qr(c_q).view(b,s,config.n_heads, self.d_rope)

        # Rotary Embeddings with proper dimenisons
        rotary_emb = self.rotary(s)
        cos = torch.cos(rotary_emb).view(1, s, 1, -1) # [1 , seq , 1 , dim]
        sin = torch.sin(rotary_emb).view(1, s, 1, -1) # [1, seq, 1, dim]

        # apply rotary embeddings
        q_rot - apply_ratary(q_rot, cos, sin)
        k_rot = apply_rotary(self.W_kr(x).view(b, s, config.n_heads, config.d_rope),
                             cos, sin)
        q = torch.cat([q_base, q_rot], dim=-1)
        k = torch.cat([k, k_rot], dim=-1)

        # Attention computation 
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bkhd->bqhd", attn, v)
        return self.output(out.contiguous().view(b,s,-1)),(c_kv, k_rot)

        

class MLA(nn.Module):
    """
    Multi-headed Attention Layer (MLA)
    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """
    def __init__(self, args:Config):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim , self.n_heads * self.qk_head_dim)
        else:
            # self.wq_a = 

        
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

# @dataclass
# class ModelArgs:
#     max_batch_size: int = 8
#     max_seq_len: int = 4096 * 4
#     dtype: Literal["bf16", "fp8"] = "bf16"
#     vocab_size: int = 102400
#     dim: int = 2048
#     inter_dim: int = 10944
#     moe_inter_dim: int = 1408
#     n_layers: int = 27
#     n_dense_layers: int = 1
#     n_heads: int = 16
#     # moe
#     n_routed_experts: int = 64
#     n_shared_experts: int = 2
#     n_activated_experts: int = 6
#     n_expert_groups: int = 1
#     n_limited_groups: int = 1
#     score_func: Literal["softmax", "sigmoid"] = "softmax"
#     route_scale: float = 1.
#     # mla
#     q_lora_rank: int = 0
#     kv_lora_rank: int = 512
#     qk_nope_head_dim: int = 128
#     qk_rope_head_dim: int = 64
#     v_head_dim: int = 128
#     # yarn
#     original_seq_len: int = 4096
#     rope_theta: float = 10000.0
#     rope_factor: float = 40
#     beta_fast: int = 32
#     beta_slow: int = 1
#     mscale: float = 1.


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
    original_seq_len:int = 4096
    rope_theta : float = 10000.0
    rope_factor : float = 40
    beta_fast : int = 32
    beta_slow : int = 1
    mscale : float = 1

def precompute_freqs_cis(args:ModelArgs)-> torch.Tensor:
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow =  args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        """
        return dim * math.log(max_seq_len / (num_rotations* 2 * math.pi)) / (2 * math.log(base))
    
    def find_correction_range(low_rot, high_rot, dim, base,max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high= math.ceil(find_correction_dim(high_rot, dim, base , max_seq_len))
        return max(low, 0), min(high, dim-1)
    
    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func
    freqs = 1.0 / (base ** (torch.arange(0, dim,2,dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low , high = find_correction_range(beta_fast,beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low , high, dim//2)
        freqs =freqs/ factor * (1 - smooth) + freqs * smooth
    t = torch.arange(seqlen)
    freqs = torch.outer(t , freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


        
class RMSNorm1(nn.Module):
    """
    Root Mean square Layer Normaalization (RMSNorm)

    """
    def __init__(self, dim:int, eps:float=1e-6):
        super().__init__()
        self.dim = dim
        self.eps= eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)
    
class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def norm(self, x:torch.Tensor):
        # rsqrt : 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self , x):
        return self.weight * self.norm(x.float()).type_as(x)
    

def apply_rotary_emb(x:torch.Tensor, freqs_cis:torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional emebdding to the input tensor

    """
    dtype = x.dtype
    x =torch.view_as_complex(x.float().view(*x.shape[:-1],-1,2))
    freqs_cis = freqs_cis.view(1,x.size(1), 1, x.shape(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


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
        self.dim = args.dim  # Input dimension
        self.n_heads = args.n_heads  # Number of attention heads
        self.n_local_heads = args.n_heads // world_size  # Heads per GPU for distributed training

        self.q_lora_rank = args.q_lora_rank  # Low-rank dimension for query projection
        self.kv_lora_rank = args.kv_lora_rank  # Low-rank dimension for key/value projection

        self.qk_nope_head_dim = args.qk_nope_head_dim  # Dimension for non-positional encoding
        self.qk_rope_head_dim = args.qk_rope_head_dim  # Dimension for rotary positional encoding
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim  # Total query/key dimension
        self.v_head_dim = args.v_head_dim  # Value dimension


        self.wq_down  = nn.Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank)
        self.wq_up = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim)

        # Key-Value projections
        self.wkv_down = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_up = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))

        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5

    def forward(self, x:torch.Tensor, start_pos:int, freqs_cis:torch.Tensor, mask:Optional[torch.Tensor]):
        """
        Forward pass for the multi headed attention Layer"""

        bsz, seqlen, _= x.size()
        end_pos = start_pos + seqlen
        q = self.wq_up(self.q_norm(self.wq_down(x)))
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q,[self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        # Key-Value compression (eq, 1-5 in paper)
        kv = self.wkv_down(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_norm(kv)
        kv = self.wkv_up(kv)
        kv = kv.view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Apply Rope to key positional component
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2),freqs_cis)

        # Combine positional and non-positional components
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)

        q2 = q.transpose(1, 2)  # [b, h, s, d]
        k2 = k.transpose(1, 2)  # [b, h, t, d]

        scores = torch.matmul(q2, k2.transpose(-2, -1)) * self.softmax_scale
        
        if mask is not None:
            scores = scores + mask.unsqueeze(1)
        scores = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(x)

        v = v.transpose(1, 2)  # [batch_size, n_heads, seq_len_k, head_dim]

        # Compute output (11 in paaper)
        output = torch.matmul(scores, v)  # [batch_size, n_heads, seq_len_q, head_dim]
        output = output.transpose(1, 2)  # [batch_size, seq_len_q, n_heads, head_dim]
        output = self.wo(output.reshape(bsz, seqlen, -1))
        return output

class ParallelEmbedding(nn.Module):
    """
    Embedding layer with parallelism support across distributed processes.

    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Embedding dimension.
    """

    def __init__(self, vocab_size:int , dim:int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim




        
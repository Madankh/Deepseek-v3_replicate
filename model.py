import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint
import math

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
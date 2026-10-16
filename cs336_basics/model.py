import torch
import torch.nn as nn
import einops
import math


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.w: torch.Tensor = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        sigma = math.sqrt(2/(in_features + out_features))
        nn.init.trunc_normal_(self.w, a=-3*sigma, b=3*sigma, std=sigma)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(self.w, x, "d_out d_in, ... d_in -> ... d_out")
    

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.embeddings = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embeddings, std=1, a=-3, b=3)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps: float = eps
        self.g = nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = (einops.reduce(x ** 2, "... d_model -> ... 1", "mean") + self.eps).sqrt()
        return (x / rms * self.g).to(in_dtype)
    

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, device=None, dtype=None):
        super().__init__()
        d_ff = math.ceil(8/3 * d_model / 64) * 64
        self.w1 = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        sigma = math.sqrt(2 / (d_ff + d_model))
        nn.init.trunc_normal_(self.w1, a=-3*sigma, b=3*sigma, std=sigma)
        nn.init.trunc_normal_(self.w2, a=-3*sigma, b=3*sigma, std=sigma)
        nn.init.trunc_normal_(self.w3, a=-3*sigma, b=3*sigma, std=sigma)
    
    @staticmethod
    def silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = einops.einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff")
        x3 = einops.einsum(self.w3, x, "d_ff d_model, ... d_model -> ... d_ff")
        return einops.einsum(self.w2, self.silu(x1) * x3, "d_model d_ff, ... d_ff -> ... d_model")
    
    
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        r_all = torch.zeros(max_seq_len, d_k, d_k, device=device)
        self.register_buffer("r_all", r_all)
        
        for i in range(max_seq_len):
            for k in range(d_k // 2):
                theta_ik = i / (theta ** (2*k/d_k))
                c = math.cos(theta_ik)
                s = math.sin(theta_ik)
                blk = torch.tensor([
                    [c, -s],
                    [s,  c]
                ], device=device)
                self.r_all[i, 2*k : 2*k + 2, 2*k : 2*k + 2] = blk
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: batch, seq_len, d_k
        # token_positions: batch, seq_len
        r = self.r_all[token_positions]
        return einops.einsum(r, x, "... max_seq_len i j, ... max_seq_len j -> ... max_seq_len i")
    

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    maxs = torch.max(x, dim, keepdim=True).values
    return torch.exp(x - maxs) / torch.sum(torch.exp(x - maxs))


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
    # q: batch, ..., seq_len, d_k
    # k: batch, ..., seq_len, d_k
    # v: batch, ..., seq_len, d_v
    # mask: seq_len, seq_len
    d_k = q.shape[-1]
    score = einops.einsum(q, k, "... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k") / math.sqrt(d_k)
    if mask is not None:
        score = torch.where(~mask, -torch.inf, score)
    return einops.einsum(softmax(score, dim=-1), v, "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v")


class CausalMultiHeadSelfAttn(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, device=None):
        super().__init__()
        # 张量维度必须是int
        self.d_k = self.d_v = d_model // num_heads
        self.num_heads = num_heads
        self.linear_wo = Linear(num_heads * self.d_v, d_model, device=device)
        self.linear_wq = Linear(d_model, num_heads * self.d_k, device=device)
        self.linear_wk = Linear(d_model, num_heads * self.d_k, device=device)
        self.linear_wv = Linear(d_model, num_heads * self.d_v, device=device)
        self.rope = RoPE(theta, self.d_k, max_seq_len, device=device)
    
    def forward(self, x: torch.Tensor,  token_positions: torch.Tensor=None) -> torch.Tensor:
        # mask: seq_len, seq_len
        seq_len = x.shape[-2]
        
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=bool))
        q = self.linear_wq(x)
        k = self.linear_wk(x)
        v = self.linear_wv(x)
        
        q = einops.rearrange(q, "... seq_len (self.num_heads d_k) -> ... self.num_heads seq_len d_k")
        k = einops.rearrange(k, "... seq_len (self.num_heads d_k) -> ... self.num_heads seq_len d_k")
        v = einops.rearrange(v, "... seq_len (self.num_heads d_v) -> ... self.num_heads seq_len d_v")
        
        # apply rope
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        
        multi_head = einops.rearrange(scaled_dot_product_attention(q, k, v, mask), "... self.num_heads seq_len d_v -> ... seq_len (self.num_heads d_v)")
        return self.linear_wo(multi_head)
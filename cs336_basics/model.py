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
    
    
class RotaryPositionalEmbedding(nn.Module):
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



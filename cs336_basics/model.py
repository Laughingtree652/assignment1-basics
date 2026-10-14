import torch
import torch.nn as nn
import einops
import math


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        in_features: int = in_features
        out_features: int = out_features
        device: torch.device | None = device
        dtype: torch.dtype | None = dtype
        super().__init__()
        self.w: torch.Tensor = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        sigma = math.sqrt(2/(in_features + out_features))
        nn.init.trunc_normal_(self.w, a=-3*sigma, b=3*sigma, std=sigma)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(self.w, x, "d_out d_in, ... d_in -> ... d_out")

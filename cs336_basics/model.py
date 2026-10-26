from collections.abc import Callable, Iterable
from typing import Optional
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
    def __init__(self, d_model: int, d_ff: int=None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
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
    # dim: 沿哪一维做（seq_len) 那维
    maxs = torch.max(x, dim, keepdim=True).values
    return torch.exp(x - maxs) / torch.sum(torch.exp(x - maxs), dim, keepdim=True)


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
        
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=bool))
        q = self.linear_wq(x)
        k = self.linear_wk(x)
        v = self.linear_wv(x)
        
        q = einops.rearrange(q, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        k = einops.rearrange(k, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        v = einops.rearrange(v, "... seq_len (h d_v) -> ... h seq_len d_v", h=self.num_heads)
        
        # apply rope
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        
        multi_head = einops.rearrange(scaled_dot_product_attention(q, k, v, mask), "... h seq_len d_v -> ... seq_len (h d_v)", h=self.num_heads)
        return self.linear_wo(multi_head)
    

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self.rmsnorm1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.rmsnorm2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.mha = CausalMultiHeadSelfAttn(d_model=d_model, num_heads=num_heads, theta=theta, max_seq_len=max_seq_len, device=device)
        self.swiglu = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        
    def forward(self, x: torch.Tensor):
        y = x + self.mha(self.rmsnorm1(x))
        return y + self.swiglu(self.rmsnorm2(y))


class TransformerLm(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, theta: float, device=None, dtype=None):
        super().__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, theta=theta, max_seq_len=context_length, device=device, dtype=dtype) for i in range(num_layers)])
        self.norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.linear = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.embedding(x)
        for _, l in enumerate(self.layers):
            y = l(y)
        return self.linear(self.norm(y))
    

def transformer_accounting(vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int):
    print("===")
    embedding_paras = vocab_size * d_model  # embedding
    blk_paras = num_layers * (d_model + d_model * d_model * 4 + d_model + d_ff * d_model * 3)
    norm_paras = d_model
    linear_paras = d_model * vocab_size
    paras = embedding_paras + blk_paras + norm_paras + linear_paras
    mem = paras * 4 / math.pow(1000, 3)
    print("trainable parameters: ", paras)
    print(f"memory required: {mem} GB")
    
    embedding_flops = 0  # embedding
    blk_linear_flops = 4 * 2 * context_length * d_model * d_model
    # attn flops: 2 * 2 * context_length * context_length * d_model // num_heads * num_heads
    blk_attn_flops = 2 * 2 * context_length * context_length * d_model 
    blk_swiglu_flops = 3 * 2 * d_ff * d_model * context_length
    blk_flops = num_layers * (blk_linear_flops + blk_attn_flops + blk_swiglu_flops)
    norm_flops = 0
    linear_flops = 2 * d_model * vocab_size * context_length
    flops = embedding_flops + blk_flops + norm_flops + linear_flops
    attn_flops = num_layers * 4 * context_length**2 * d_model
    attn_flops_pct = attn_flops / flops
    print("total flops: ", flops)
    print("attn_flops: ", attn_flops)
    print("attn_flops_pct: ", attn_flops_pct)
    estimate = 2 * paras * context_length
    print("estimate: ", estimate)
    print("flops - estimate: ", flops - estimate)
    return flops


def cross_entropy(o: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # o: ..., vocab_size
    # x: ...
    maxs = torch.max(o, dim=-1, keepdim=True).values
    y = x.unsqueeze(-1)
    o_x = torch.gather(o, dim=-1, index=y)
    return torch.mean(torch.log(torch.sum(torch.exp(o - maxs), dim=-1, keepdim=True)) - o_x + maxs)


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


def training_loop(lr):
    print("===")
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    for t in range(10):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if min(betas) < 0 or max(betas) >= 1:
            raise ValueError(f"Invalid betas: {betas}")
        if eps <= 0:
            raise ValueError(f"Invalid eps: {eps}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                t = t + 1
                lr_t = lr * math.sqrt(1 - betas[1]**t) / (1 - betas[0]**t)
                p.data.add_(- lr * weight_decay * p.data)
                if "m" not in state:
                    m = state["m"] = torch.zeros_like(p)
                    v = state["v"] = torch.zeros_like(p)
                else:
                    m = state["m"]
                    v = state["v"]
                m.mul_(betas[0]).add_(grad, alpha=1-betas[0])
                v.mul_(betas[1]).add_(grad**2, alpha=1-betas[1])
                p.data.add_(- lr_t * m / (torch.sqrt(v) + eps))
                state["t"] = t
        return loss

    @staticmethod
    def accounting(batch_size, vocab_size, context_length, num_layers, d_model, num_heads):
        paras = (2 *vocab_size + 1) * d_model + num_layers * (2 * d_model +  12 * d_model**2)
        grads = paras
        opt_states = 2 * paras
        activations = num_layers * (56/3 * batch_size * context_length * d_model + 2 * batch_size * num_heads * context_length**2) + batch_size * context_length * d_model + 2 * batch_size * context_length * vocab_size
        
        mem_paras = 4 * paras
        mem_grads = 4 * grads
        mem_opt_states = 4 * opt_states
        mem_activations = 4 * activations
        total = mem_paras + mem_grads + mem_opt_states + mem_activations
        print("===")
        print("mem para: ", mem_paras)
        print("mem_grads: ", mem_grads)
        print("mem_opt_states: ", mem_opt_states)
        print("mem_activations: ", mem_activations)
        print("total: ", total)
        
        one_step_adam_flops = paras * 14
        total_flops = (3 * batch_size * transformer_accounting(vocab_size, context_length, num_layers, d_model, num_heads, 8 * d_model // 3) + one_step_adam_flops) * 4e5
        secs = total_flops / (495e12 * 0.5)
        print("secs: ", secs)
        print("days: ", secs / (3600*24))
        
        
def learning_rate_schedule(t: int, lr_max: float, lr_min: float, t_w: int, t_c: int):
    if t < t_w:
        return t / t_w * lr_max
    elif t_w <= t <= t_c:
        return lr_min + 1/2 * (1 + math.cos((t-t_w)*math.pi/(t_c-t_w)))*(lr_max - lr_min)
    else:
        return lr_min
    
    
def gradient_clipping(paras, max_l2_norm: float, eps=1e-6):
    norm_sum = 0
    paras_list = list(paras)
    for p in paras_list:
        if p.grad is None:
            continue
        g = p.grad
        norm = torch.norm(g)
        norm_sum += norm**2
    
    l2_norm_sum = torch.sqrt(norm_sum)
    if l2_norm_sum > max_l2_norm:
        for p in paras_list:
            if p.grad is None:
                continue
            g = p.grad
            g.mul_(max_l2_norm / (l2_norm_sum + eps))


if __name__ == "__main__":
    # transformer_accounting(50257, 1024, 48, 1600, 25, 4288)
    # transformer_accounting(50257, 16384, 48, 1600, 25, 4288)
    # training_loop(1)
    # training_loop(1e2)
    # training_loop(1e3)
    AdamW.accounting(1, 50257, 1024, 48, 1600, 25)
    AdamW.accounting(2, 50257, 1024, 48, 1600, 25)
    AdamW.accounting(1024, 50257, 1024, 48, 1600, 25)
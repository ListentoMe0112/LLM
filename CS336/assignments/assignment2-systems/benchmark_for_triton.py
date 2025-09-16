from utils import FlashAttentionTriton, FlashAttention
import torch
import triton
import cs336_basics
from cs336_basics.utils import TransformerLanguageModel, cross_entropy, softmax, AdamW

import torch
import triton
import triton.testing as testing
import itertools
import pandas as pd
import os

# Import your autograd.Function classes
# Expected to be defined as:
#   class FlashAttn2Triton(torch.autograd.Function):
#   class FlashAttn2Torch(torch.autograd.Function):
try:
    from utils import FlashAttention, FlashAttentionTriton
except ImportError:
    raise ImportError("Please provide your autograd.Function classes for Triton and Torch implementations.")


# Wrappers around the autograd.Functions
def fa2_triton(q, k, v, causal=True):
    return FlashAttention.apply(q, k, v, causal)

def fa2_torch(q, k, v, causal=True):
    return FlashAttentionTriton.apply(q, k, v, causal)


# Ensure tensors are created on the correct device/dtype
def run_forward(func, q, k, v):
    return func(q, k, v, True)

def run_backward(func, q, k, v):
    device, dtype = q.device, q.dtype
    q = q.clone().detach().requires_grad_(True).to(device=device, dtype=dtype)
    k = k.clone().detach().requires_grad_(True).to(device=device, dtype=dtype)
    v = v.clone().detach().requires_grad_(True).to(device=device, dtype=dtype)
    o = func(q, k, v, True)
    dO = torch.randn_like(o, device=device, dtype=dtype)
    o.backward(dO, retain_graph=True)
    return q.grad, k.grad, v.grad

def run_end2end(func, q, k, v):
    device, dtype = q.device, q.dtype
    q = q.clone().detach().requires_grad_(True).to(device=device, dtype=dtype)
    k = k.clone().detach().requires_grad_(True).to(device=device, dtype=dtype)
    v = v.clone().detach().requires_grad_(True).to(device=device, dtype=dtype)
    o = func(q, k, v, True)
    dO = torch.randn_like(o, device=device, dtype=dtype)
    o.backward(dO)
    return o


# Benchmark grid
seq_lens = [2**i for i in range(7, 17)]  # 128 to 65536
embed_dims = [16, 32, 64, 128]
dtypes = [torch.bfloat16, torch.float32]

results = []

for S, D, dtype in itertools.product(seq_lens, embed_dims, dtypes):
    device = "cuda"
    B, H = 1, 1  # batch=1, single head for simplicity

    q = torch.randn((B, S, H, D), device=device, dtype=dtype, requires_grad=True)
    k = torch.randn((B, S, H, D), device=device, dtype=dtype, requires_grad=True)
    v = torch.randn((B, S, H, D), device=device, dtype=dtype, requires_grad=True)

    for name, func in [
        ("Triton", fa2_triton),
        ("Torch", fa2_torch),
    ]:
        try:
            fwd_ms = testing.do_bench(lambda: run_forward(func, q, k, v))
            bwd_ms = testing.do_bench(lambda: run_backward(func, q, k, v))
            e2e_ms = testing.do_bench(lambda: run_end2end(func, q, k, v))

            results.append({
                "impl": name,
                "seq_len": S,
                "embed_dim": D,
                "dtype": str(dtype),
                "forward_ms": fwd_ms,
                "backward_ms": bwd_ms,
                "end2end_ms": e2e_ms,
            })
        except RuntimeError as e:
            results.append({
                "impl": name,
                "seq_len": S,
                "embed_dim": D,
                "dtype": str(dtype),
                "forward_ms": None,
                "backward_ms": None,
                "end2end_ms": None,
                "error": str(e),
            })


# Save results
os.makedirs("bench_results", exist_ok=True)
df = pd.DataFrame(results)
df.to_csv("bench_results/fa2_triton_vs_torch.csv", index=False)

with open("bench_results/fa2_triton_vs_torch.md", "w") as f:
    f.write(df.to_markdown())

print("Benchmark finished. Results saved to bench_results/fa2_triton_vs_torch.*")


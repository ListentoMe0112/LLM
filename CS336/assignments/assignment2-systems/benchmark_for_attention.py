from __future__ import annotations
import cs336_basics
from cs336_basics.utils import TransformerLanguageModel, cross_entropy, softmax, AdamW
import argparse
import torch
import timeit
import functools
import torch.cuda.nvtx as nvtx
from torch import Tensor

from einops import einsum, reduce,rearrange, repeat
from jaxtyping import Float, Int

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(q: Float[Tensor, " ... queries d_k"],
    k: Float[Tensor, " ... keys d_k"],
    v: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    with nvtx.range("computing attention scores"):
        inv_d_k = torch.sqrt(torch.tensor(1 / k.shape[-1], dtype = torch.float32))
        pre_softmax_score = einsum(q, k, "... queries d_k, ... keys d_k -> ... queries keys")
        pre_softmax_score *= inv_d_k

        if mask != None:
            mask = mask.to(torch.float32)
            mask = ((1 - mask) * 1e9)
            pre_softmax_score = pre_softmax_score - mask

    with nvtx.range("computing softmax"):
        score = softmax(pre_softmax_score, dim = -1)
    with nvtx.range("final matmul"):
        attention = einsum(score, v, "... queries keys, ... keys d_v -> ... queries d_v")
    return attention

cs336_basics.utils.scaled_dot_product_attention = annotated_scaled_dot_product_attention

compiled_attention = torch.compile(annotated_scaled_dot_product_attention)

@nvtx.range("forward_and_backward")
def one_step(model : torch.nn.Module, input: torch.Tensor, output: torch.tensor, optimizer):
    with nvtx.range("forward"):
        logits = model(dummy_input)

    with nvtx.range("loss_compute"):
        loss = cross_entropy(logits.view(-1, logits.size(-1)), dummy_output.view(-1))

    with nvtx.range("optimizer_zero"):
        optimizer.zero_grad(set_to_none=True)

    with nvtx.range("backward"):
        loss.backward()

    with nvtx.range("optimizer_step"):
        optimizer.step()
        
    torch.cuda.synchronize()

@nvtx.range("forward")
def forward_time(model : torch.nn.Module, input: torch.Tensor):
    logits = model(dummy_input)
    torch.cuda.synchronize()
    
@nvtx.range("backward")
def backward_time(logits):
    logits.backward()
    torch.cuda.synchronize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer Language Model Training')
    parser.add_argument('--compile', action="store_true",help="compile")
    parser.add_argument('--no_compile', action="store_false", dest="compile",  help="compile")
    args = parser.parse_args()
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]
    batch_size = 8
    num_heads = 1  # Disable multi-head attention
    vocab_size = 50527
    warm_up_iters = 3
    test_iters = 100

    results = []
    
    for d_model in d_models:
        for seq_len in seq_lens:
            try:
                # Create inputs with current configuration
                q_tensor = torch.rand(8, seq_len, d_model, requires_grad=True).to('cuda')
                k_tensor = torch.rand(8, seq_len, d_model, requires_grad=True).to('cuda')
                v_tensor = torch.rand(8, seq_len, d_model, requires_grad=True).to('cuda')
                
                # Warm-up phase
                for _ in range(warm_up_iters):
                    output = annotated_scaled_dot_product_attention(q_tensor, k_tensor, v_tensor)
                    loss = output.sum()
                    loss.backward()
                    torch.cuda.synchronize()
                
                # Forward pass timing
                forward_times = []
                backward_times = []
                memory_before_backward_list = []
                for _ in range(test_iters):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    if args.compile:
                        output = compiled_attention(q_tensor, k_tensor, v_tensor)
                    else:
                        output = annotated_scaled_dot_product_attention(q_tensor, k_tensor, v_tensor)
                    end.record()
                    torch.cuda.synchronize()
                    forward_times.append(start.elapsed_time(end))
                
                    # Memory measurement before backward
                    memory_before_backward = torch.cuda.memory_allocated()
                    loss = output.sum()

                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    loss.backward()
                    memory_used = torch.cuda.memory_allocated()
                    memory_before_backward_list.append(memory_used - memory_before_backward)
                    end.record()
                    torch.cuda.synchronize()
                    backward_times.append(start.elapsed_time(end))
                
                # Store results
                results.append({
                    'd_model': d_model,
                    'seq_len': seq_len,
                    'forward_avg': sum(forward_times)/len(forward_times),
                    'backward_avg': sum(backward_times)/len(backward_times),
                    'memory_usage_before_backward': (sum(memory_before_backward_list) / len(memory_before_backward_list)) / (1024**2),
                    'oom': False
                })
                
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    results.append({
                        'd_model': d_model,
                        'seq_len': seq_len,
                        'oom': True
                    })
                    torch.cuda.empty_cache()
                else:
                    raise

    # Print results in a table format
    print(f"{'d_model':<8} | {'seq_len':<8} | {'forward (ms)':<12} | {'backward (ms)':<12} | {'memory (MB)':<12} | OOM")
    print('-' * 70)
    for res in results:
        if res['oom']:
            print(f"{res['d_model']:<8} | {res['seq_len']:<8} | {'-':<12} | {'-':<12} | {'-':<12} | Yes")
        else:
            print(f"{res['d_model']:<8} | {res['seq_len']:<8} | {res['forward_avg']:<12.2f} | {res['backward_avg']:<12.2f} | {res['memory_usage_before_backward']:<12.2f} | No")

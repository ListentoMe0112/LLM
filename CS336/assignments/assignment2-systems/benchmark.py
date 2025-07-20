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

# small 768 3072 12 12
# medium 1024 4096 24 16
# large 1280 5120 36 20
# xl 1600 6400 48 25
# 2.7B 2560 10240 32 32

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
    parser.add_argument('--d_model', type=int, required=True, help="d_model")
    parser.add_argument('--d_ff', type=int, required=True, help="d_ff")
    parser.add_argument('--num_layers', type=int, required=True, help="num_layers")
    parser.add_argument('--num_heads', type=int, required=True, help="num_heads")
    parser.add_argument('--run_backward', type=bool, required=True, help="run_backward")
    parser.add_argument('--warm_up', type=bool, required=True, help="warm_up")
    parser.add_argument('--iteration', type=bool, required=True, help="warm_up")
    args = parser.parse_args()

    dummy_input = torch.randint(0, 50527, (4, 1024)).to('cuda')
    dummy_output = torch.randint(0, 50527, (4,1024)).to('cuda')
    model = TransformerLanguageModel(50527, args.d_model, args.num_heads, 10000.0, 1024, args.d_ff, args.num_layers, device='cuda').to('cuda')


    optimizer = AdamW(
        model.parameters(),
        lr=0.01,
        weight_decay=0.99,
        betas=(0.9, 0.95),
        eps=1e-8
    )

    for i in range(args.warm_up):
        one_step(model, dummy_input, dummy_output,optimizer)
        torch.cuda.synchronize()

    partial_function = functools.partial(one_step, model, dummy_input, dummy_output, optimizer)
    time_taken = timeit.timeit(stmt=partial_function, number=args.iteration)
    print(f"Execution time with forward and backward: {time_taken}")

    # partial_function = functools.partial(forward_time, model, dummy_input)
    # time_taken = timeit.timeit(stmt=partial_function, number=args.iteration)
    # print(f"Execution time with forward: {time_taken}")

    # logits = model(dummy_input)
    # loss = cross_entropy(logits.view(-1, logits.size(-1)), dummy_output.view(-1))
    # partial_function = functools.partial(backward_time, loss)
    # time_taken = timeit.timeit(stmt=partial_function, number=args.iteration)
    # print(f"Execution time with backward: {time_taken}")


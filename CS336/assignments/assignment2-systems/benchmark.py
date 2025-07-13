import cs336_basics
from cs336_basics.utils import TransformerLanguageModel, cross_entropy
import argparse
import torch
import timeit
import functools

def forward_and_backward_time(model : torch.nn.Module, input: torch.Tensor, output: torch.tensor):
    logits = model(dummy_input)
    loss = cross_entropy(logits.view(-1, logits.size(-1)), dummy_output.view(-1))
    loss.backward()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer Language Model Training')
    parser.add_argument('--d_model', type=int, required=True, help="d_model")
    parser.add_argument('--d_ff', type=int, required=True, help="d_ff")
    parser.add_argument('--num_layers', type=int, required=True, help="num_layers")
    parser.add_argument('--num_heads', type=int, required=True, help="num_heads")
    parser.add_argument('--run_backward', type=bool, required=True, help="run_backward")
    parser.add_argument('--warm_up', type=bool, required=True, help="warm_up")
    args = parser.parse_args()

    dummy_input = torch.randint(0, 50527, (4, 1024))
    dummy_output = torch.randint(0, 50527, (4,1024))
    model = TransformerLanguageModel(50527, args.d_model, args.num_heads, 10000.0, 1024, args.d_ff, args.num_layers)
    for i in range(args.warm_up):
        forward_and_backward_time(model, dummy_input, dummy_output)

    print("warm up finished")

    partial_function = functools.partial(forward_and_backward_time, model, dummy_input, dummy_output)
    time_taken = timeit.timeit(stmt=partial_function, number=10)
    print(f"Execution time with functools.partial: {time_taken}")


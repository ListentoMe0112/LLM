# Write a script to naively perform distributed data parallel training by all-reducing
# individual parameter gradients after the backward pass. To verify the correctness of your DDP implementation,
# use it to train a small toy model on randomly-generated data and verify that its weights
# match the results from single-process training

import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.cuda.nvtx as nvtx
from copy import deepcopy
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from cs336_basics.utils import TransformerLanguageModel

class DistributedDataParallel:
    def __init__(self, model):
        self.model = model
        self.params = list(model.parameters())
        self.rank = dist.get_rank()
        self.module = self.model
        
        # Broadcast parameters from rank 0 to all processes
        with torch.no_grad():
            if self.rank == 0:
                param_buffer = [p.detach() for p in self.params]
            else:
                param_buffer = [torch.zeros_like(p) for p in self.params]
            
            for p in param_buffer:
                dist.broadcast(p, src=0)
            
            for model_p, buffer_p in zip(self.params, param_buffer):
                model_p.data.copy_(buffer_p)

    def named_parameters(self):
        return self.model.named_parameters()
    
    def parameters(self):
        return self.model.parameters()
        
    def all_reduce_gradients(self):
        """All-reduce gradients across processes"""
        comms = []
        for param in self.params:
            if param.requires_grad and param.grad is not None:
                comms.append(param.grad)
        flat = _flatten_dense_tensors(comms)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat /= dist.get_world_size()
        grads = _unflatten_dense_tensors(flat, comms)
        for g, p in zip(grads, comms):
            p.copy_(g)

    def __call__(self, x):
        return self.model(x)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_process(rank, world_size):
    setup(rank, world_size)
    device = f'cuda:{rank}'
    dist.barrier()

    torch.manual_seed(rank)
    
    # Model setup
    non_parallel_model = TransformerLanguageModel(
        50527, 1600, 25, 10000.0, 128, 6400, 48, device=device
    ).to(device)
    ddp_base = deepcopy(non_parallel_model)
    ddp_model = DistributedDataParallel(ddp_base)

    # Data preparation
    all_x = torch.randint(0, 50527, (8, 128)).to(device)
    all_y = torch.randint(0, 50527, (8, 128)).to(device)
    all_y = torch.nn.functional.one_hot(all_y, num_classes=50527).float()

    assert all_x.size(0) % world_size == 0
    local_bs = int(all_y.size(0) / world_size)

    loss_fn = nn.CrossEntropyLoss()
    ddp_optimizer = optim.SGD(ddp_model.parameters(), lr=0.1)

    # Benchmark parameters
    total_steps = 100
    total_step_time = 0.0
    total_comm_time = 0.0
    
    # Warmup iterations
    for _ in range(3):
        ddp_optimizer.zero_grad()
        ddp_outputs = ddp_model(all_x[:local_bs])
        ddp_loss = loss_fn(ddp_outputs, all_y[:local_bs])
        ddp_loss.backward()
        ddp_model.all_reduce_gradients()
        ddp_optimizer.step()

    # Start profiling
    if rank == 0:
        torch.cuda.cudart().cudaProfilerStart()
    
    for i in range(total_steps):
        nvtx.range_push(f"Step {i}")
        ddp_optimizer.zero_grad()
        
        # Data sharding
        offset = rank * local_bs
        ddp_data = all_x[offset:offset+local_bs]
        ddp_labels = all_y[offset:offset+local_bs]

        # CUDA events setup
        if rank == 0 and i == 0:
            step_start_event = torch.cuda.Event(enable_timing=True)
            step_end_event = torch.cuda.Event(enable_timing=True)
            comm_start_event = torch.cuda.Event(enable_timing=True)
            comm_end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        if rank == 0:
            step_start_event.record()

        # Forward pass
        nvtx.range_push("Forward")
        ddp_outputs = ddp_model(ddp_data)
        ddp_loss = loss_fn(ddp_outputs, ddp_labels)
        torch.cuda.synchronize()
        nvtx.range_pop()

        # Backward pass
        nvtx.range_push("Backward")
        ddp_loss.backward()
        torch.cuda.synchronize()
        nvtx.range_pop()

        # Gradient synchronization
        torch.cuda.synchronize()
        if rank == 0:
            comm_start_event.record()
        
        nvtx.range_push("Gradient Sync")
        ddp_model.all_reduce_gradients()
        torch.cuda.synchronize()
        nvtx.range_pop()
        
        torch.cuda.synchronize()
        if rank == 0:
            comm_end_event.record()

        # Optimization step
        nvtx.range_push("Optimizer Step")
        ddp_optimizer.step()
        torch.cuda.synchronize()
        nvtx.range_pop()

        torch.cuda.synchronize()
        nvtx.range_pop()  # End step range

        # Timing collection
        if rank == 0:
            step_end_event.record()
            torch.cuda.synchronize()
            
            comm_time = comm_start_event.elapsed_time(comm_end_event) / 1000
            step_time = step_start_event.elapsed_time(step_end_event) / 1000
            
            total_comm_time += comm_time
            total_step_time += step_time

        # Data shuffling
        shuffle_idxs = torch.randperm(all_x.size(0))
        all_x = all_x[shuffle_idxs]
        all_y = all_y[shuffle_idxs]

    if rank == 0:
        avg_step_time = total_step_time / total_steps
        comm_proportion = total_comm_time / total_step_time
        print(f"\nBenchmark Results (averaged over {total_steps} steps):")
        print(f"Total time per training step: {avg_step_time:.4f} seconds")
        print(f"Proportion of time spent on communication: {comm_proportion:.2%}")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(train_process, args=(world_size,), nprocs=world_size, join=True)

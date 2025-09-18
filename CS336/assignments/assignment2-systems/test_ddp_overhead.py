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
from copy import deepcopy

from tests.adapters import (
    ddp_individual_parameters_on_after_backward,
    get_ddp_individual_parameters,
)
from tests.common import (
    FIXTURES_PATH,
    ToyModel,
    ToyModelWithTiedWeights,
    _cleanup_process_group,
    _setup_process_group,
    validate_ddp_net_equivalence,
)

from cs336_basics.utils import (
    TransformerLanguageModel
)

# class ToyModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(10, 5)
#         self.fc2 = nn.Linear(5, 5)
#     
#     def forward(self, x):
#         return self.fc2(torch.relu(self.fc1(x)))

class DistributedDataParallel:
    def __init__(self, model):
        self.model = model
        self.params = list(model.parameters())
        self.rank = dist.get_rank()
        self.module = self.model
        
        # Broadcast parameters from rank 0 to all processes
        with torch.no_grad():
            # Prepare parameter buffers
            if self.rank == 0:
                param_buffer = [p.detach() for p in self.params]  # Keep parameters on GPU
            else:
                param_buffer = [torch.zeros_like(p) for p in self.params]
            
            # Broadcast each parameter
            for p in param_buffer:
                dist.broadcast(p, src=0)
            
            # Update model parameters with broadcasted values
            for model_p, buffer_p in zip(self.params, param_buffer):
                model_p.data.copy_(buffer_p)

    def named_parameters(self):
        return self.model.named_parameters()
    
    def parameters(self):
        return self.model.parameters()
        
    def all_reduce_gradients(self):
        """All-reduce gradients across processes"""
        for param in self.params:
            if param.requires_grad and param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= dist.get_world_size()  # Average gradients

    def __call__(self, x):
        return self.model(x)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)  # Explicitly specify device_id

def cleanup():
    dist.destroy_process_group()

def train_process(rank, world_size):
    # Initialize process group
    setup(rank, world_size)
    device = f'cuda:{rank}'  # Assign different GPUs to different ranks
    dist.barrier(device_ids=[rank])  # Explicitly specify device for barrier

    # Seed to ensure that ranks are initialized with different initial models.
    torch.manual_seed(rank)
    
    # Create models and move to GPU
    non_parallel_model = TransformerLanguageModel(50527, 1600, 25, 10000.0, 128, 6400, 48, device=device).to(device)
    ddp_base = deepcopy(non_parallel_model)
    ddp_model = DistributedDataParallel(ddp_base)

    all_x = torch.randint(0, 50527, (8, 128)).to(device)
    # Convert labels to one-hot encoding
    all_y = torch.randint(0, 50527, (8, 128)).to(device)
    all_y = torch.nn.functional.one_hot(all_y, num_classes=50527).float()

    assert all_x.size(0) % world_size == 0
    local_bs = int(all_y.size(0) / world_size)

    loss_fn = nn.CrossEntropyLoss()

    # Optimizer for the DDP model
    ddp_optimizer = optim.SGD(ddp_model.parameters(), lr=0.1)

    # Benchmarking parameters
    total_steps = 100
    total_step_time = 0.0
    total_comm_time = 0.0
    
    for i in range(total_steps):
        ddp_optimizer.zero_grad()
        offset = rank * local_bs
        ddp_data = all_x[offset : offset + local_bs, :].to(device)
        ddp_labels = all_y[offset : offset + local_bs, :].to(device)
        # Create CUDA events only on rank 0
        if rank == 0 and i == 0:
            torch.cuda.set_device(device)
            step_start_event = torch.cuda.Event(enable_timing=True)
            step_end_event = torch.cuda.Event(enable_timing=True)
            comm_start_event = torch.cuda.Event(enable_timing=True)
            comm_end_event = torch.cuda.Event(enable_timing=True)

        # Synchronize before starting timing
        torch.cuda.synchronize(device=device)
        
        # Only rank 0 handles timing
        if rank == 0:
            # Record full step start
            step_start_event.record()
        
        # Forward pass
        ddp_outputs = ddp_model(ddp_data)
        ddp_loss = loss_fn(ddp_outputs, ddp_labels)
        
        # Backward pass
        ddp_loss.backward()
        
        # Synchronize before communication timing
        torch.cuda.synchronize(device=device)
        if rank == 0:
            comm_start_event.record()
        
        ddp_model.all_reduce_gradients()
        
        torch.cuda.synchronize(device=device)
        if rank == 0:
            comm_end_event.record()
        
        # Optimization step
        ddp_optimizer.step()
        
        # Synchronize before recording end event
        torch.cuda.synchronize(device=device)
        if rank == 0:
            step_end_event.record()
        
        # Calculate elapsed times only on rank 0
        if rank == 0:
            torch.cuda.synchronize(device=device)
            comm_time = comm_start_event.elapsed_time(comm_end_event) / 1000
            step_time = step_start_event.elapsed_time(step_end_event) / 1000
            
            # Accumulate timings
            total_comm_time += comm_time
            total_step_time += step_time

        dist.barrier()

        torch.manual_seed(42 + i)
        shuffle_idxs = torch.randperm(all_x.size(0))
        # 验证所有进程的洗牌索引是否一致
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
    # Run distributed training
    world_size = 2
    mp.spawn(train_process, args=(world_size,), nprocs=world_size, join=True)
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

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 5)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

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
    dist.init_process_group("nccl", rank=rank, world_size=world_size)  # Use NCCL backend for GPU training

def cleanup():
    dist.destroy_process_group()

def train_process(rank, world_size):
    # Initialize process group
    setup(rank, world_size)
    device = f'cuda:{rank}'  # Assign different GPUs to different ranks
    dist.barrier()

    # Seed to ensure that ranks are initialized with different initial models.
    torch.manual_seed(rank)
    
    # Create models and move to GPU
    non_parallel_model = ToyModel().to(device)
    ddp_base = deepcopy(non_parallel_model)
    ddp_model = DistributedDataParallel(ddp_base)

    for (non_parallel_param_name, non_parallel_model_parameter), (
        ddp_model_param_name,
        ddp_model_parameter,
    ) in zip(non_parallel_model.named_parameters(), ddp_model.named_parameters()):
        if rank == 0 :
            assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)
        else:
            assert not torch.allclose(non_parallel_model_parameter, ddp_model_parameter)

    # Make sure all the ranks have the same model state
    validate_ddp_net_equivalence(ddp_model)

    # Load the dataset from disk, so we can ensure that every rank has the same
    # overall pool of data.
    # Shape: (20, 10)
    all_x = torch.load(FIXTURES_PATH / "ddp_test_data.pt")
    # Shape: (20, 5)
    all_y = torch.load(FIXTURES_PATH / "ddp_test_labels.pt")

    assert all_x.size(0) % world_size == 0
    local_bs = int(all_y.size(0) / world_size)

    loss_fn = nn.MSELoss()

    # Optimizer for the DDP model
    ddp_optimizer = optim.SGD(ddp_model.parameters(), lr=0.1)
    # Optimizer for the non-parallel model
    non_parallel_optimizer = optim.SGD(non_parallel_model.parameters(), lr=0.1)

    # Benchmarking parameters
    total_steps = 100
    total_step_time = 0.0
    total_comm_time = 0.0
    
    for i in range(total_steps):
        ddp_optimizer.zero_grad()
        non_parallel_optimizer.zero_grad()

        # Run the non-parallel model on all the data and take a gradient step
        non_parallel_data = all_x.to(device)
        non_parallel_labels = all_y.to(device)
        non_parallel_outputs = non_parallel_model(non_parallel_data)
        non_parallel_loss = loss_fn(non_parallel_outputs, non_parallel_labels)
        non_parallel_loss.backward()
        non_parallel_optimizer.step()

        # At this point, the parameters of non-parallel model should differ
        # from the parameters of the DDP model (since we've applied the
        # gradient step to the non-parallel model, but not to the DDP model).
        if rank == 0:
            for non_parallel_model_parameter, ddp_model_parameter in zip(
                non_parallel_model.parameters(), ddp_model.parameters()
            ):
                if non_parallel_model_parameter.requires_grad and ddp_model_parameter.requires_grad:
                    # The only parameters that change are those that require_grad
                    assert not torch.allclose(non_parallel_model_parameter, ddp_model_parameter)
                else:
                    # parameters that don't require_grad shouldn't change
                    assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)

        # While the non-parallel model does a forward pass on all the data (20 examples),
        # each DDP rank only sees 10 (disjoint) examples.
        # However, the end result should be the same as doing a forward pass on all 20 examples.
        offset = rank * local_bs
        ddp_data = all_x[offset : offset + local_bs, :].to(device)
        ddp_labels = all_y[offset : offset + local_bs, :].to(device)
        # Create CUDA events for timing
        step_start_event = torch.cuda.Event(enable_timing=True)
        step_end_event = torch.cuda.Event(enable_timing=True)
        comm_start_event = torch.cuda.Event(enable_timing=True)
        comm_end_event = torch.cuda.Event(enable_timing=True)

        # Synchronize before starting timing
        torch.cuda.synchronize(device=device)
        
        # Record full step start
        step_start_event.record()
        
        # Forward pass
        ddp_outputs = ddp_model(ddp_data)
        ddp_loss = loss_fn(ddp_outputs, ddp_labels)
        
        # Backward pass
        ddp_loss.backward()
        
        # Synchronize before communication timing
        torch.cuda.synchronize(device=device)
        comm_start_event.record()
        ddp_model.all_reduce_gradients()
        torch.cuda.synchronize(device=device)  # Ensure all_reduce completes
        comm_end_event.record()
        
        # Optimization step
        ddp_optimizer.step()
        
        # Synchronize before recording end event
        torch.cuda.synchronize(device=device)
        
        # Record full step end after synchronization
        step_end_event.record()
        
        # Calculate elapsed times
        comm_time = comm_start_event.elapsed_time(comm_end_event) / 1000  # Convert ms to seconds
        step_time = step_start_event.elapsed_time(step_end_event) / 1000  # Convert ms to seconds
        
        # Accumulate timings
        total_comm_time += comm_time
        total_step_time += step_time

        dist.barrier()

        # At this point, the non-parallel model should exactly match the parameters of the DDP model
        if rank == 0:
            for non_parallel_model_parameter, ddp_model_parameter in zip(
                non_parallel_model.parameters(), ddp_model.parameters()
            ):
                assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter, rtol=1e-4, atol=1e-7), f"{i} trigger not close"

        # Shuffle the data so that during the next iteration, each DDP rank sees a different set of inputs.
        # We make sure to use the same seed when shuffling (else the per-rank examples might not be disjoint).
        torch.manual_seed(42 + i)
        shuffle_idxs = torch.randperm(all_x.size(0))
        # 验证所有进程的洗牌索引是否一致
        all_x = all_x[shuffle_idxs]
        all_y = all_y[shuffle_idxs]

    # Print benchmark results from rank 0
    if rank == 0:
        for non_parallel_model_parameter, ddp_model_parameter in zip(
            non_parallel_model.parameters(), ddp_model.parameters()
        ):
            assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)

    
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
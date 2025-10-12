import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.optim import Optimizer
from typing import Any, Type, List

class DistributedDataParallel(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        """
        Distributed data parallel wrapper for PyTorch modules.
        Handles parameter synchronization and gradient averaging.
        """
        super().__init__()
        self.module = module
        self.params_list = list(module.parameters())
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        # Broadcast parameters from rank 0 to all processes
        self._broadcast_initial_parameters()
        
        # Register gradient hooks
        self._register_gradient_hooks()

    def _broadcast_initial_parameters(self):
        """Broadcast parameters from rank 0 to all processes"""
        with torch.no_grad():
            # Prepare parameter buffers
            if self.rank == 0:
                param_buffer = [p.cpu().detach() for p in self.params_list]
            else:
                param_buffer = [torch.zeros_like(p.cpu()) for p in self.params_list]
            
            # Broadcast each parameter
            for p in param_buffer:
                dist.broadcast(p, src=0)
            
            # Update parameters while preserving device placement
            for model_p, buffer_p in zip(self.params_list, param_buffer):
                model_p.data.copy_(buffer_p.to(model_p.device))
    
    def parameters(self):
        return self.module.parameters()

    def _register_gradient_hooks(self):
        """Register gradient hooks for automatic synchronization"""
        for param in self.params_list:
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._gradient_synchronization_hook)

    def _gradient_synchronization_hook(self, param):
        """Hook function for gradient synchronization"""
        if param.grad is not None:
            # Flatten and average gradients
            flat_grad = _flatten_dense_tensors([param.grad])
            dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
            flat_grad /= self.world_size
            
            # Unflatten and update gradients
            unflat_grads = _unflatten_dense_tensors(flat_grad, [param.grad])
            param.grad.data.copy_(unflat_grads[0])

    def forward(self, *inputs, **kwargs):
        """Forward pass through wrapped module"""
        return self.module(*inputs, **kwargs)

class DistributedDataParallelBucket(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float = 25):
        """
        Distributed data parallel wrapper for PyTorch modules.
        Handles parameter synchronization and gradient averaging.
        """
        super().__init__()
        self.module = module
        self.params_list = list(module.parameters())
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        # Broadcast parameters from rank 0 to all processes
        self._broadcast_initial_parameters()

        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        
        # Register gradient hooks with bucketing
        self._register_bucketed_gradient_hooks()

    def _broadcast_initial_parameters(self):
        """Broadcast parameters from rank 0 to all processes"""
        with torch.no_grad():
            # Prepare parameter buffers
            if self.rank == 0:
                param_buffer = [p.cpu().detach() for p in self.params_list]
            else:
                param_buffer = [torch.zeros_like(p.cpu()) for p in self.params_list]
            
            # Broadcast each parameter
            for p in param_buffer:
                dist.broadcast(p, src=0)
            
            # Update parameters while preserving device placement
            for model_p, buffer_p in zip(self.params_list, param_buffer):
                model_p.data.copy_(buffer_p.to(model_p.device))
    
    def parameters(self):
        return self.module.parameters()

    def _register_bucketed_gradient_hooks(self):
        """Register gradient hooks with bucketing using reverse parameter order"""
        # Organize parameters into buckets based on reverse order and size
        current_bucket = []
        current_size = 0
        bucket_id = 0
        
        # Iterate parameters in reverse order (as in model.parameters())
        for param in reversed(self.params_list):
            if not param.requires_grad:
                continue
                
            param_size = param.numel() * param.element_size()
            
            # Check if parameter fits in current bucket or bucket is empty
            if current_size > 0 and current_size + param_size > self.bucket_size_bytes:
                # Register hook for current bucket
                self._register_bucket_hook(current_bucket, bucket_id)
                # Start new bucket
                current_bucket = []
                current_size = 0
                bucket_id += 1
                
            current_bucket.append(param)
            current_size += param_size
            
        # Register hook for the last bucket
        if current_bucket:
            self._register_bucket_hook(current_bucket, bucket_id)

    def _register_bucket_hook(self, bucket_params, bucket_id):
        """Register synchronization hook for a bucket of parameters"""
        def bucket_hook(_):
            # Collect gradients from all parameters in the bucket
            grads = [p.grad for p in bucket_params if p.grad is not None]
            if not grads:
                return
                
            # Flatten and average all gradients in the bucket
            flat_grad = _flatten_dense_tensors(grads)
            dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
            flat_grad /= self.world_size
            
            # Unflatten and update gradients
            unflat_grads = _unflatten_dense_tensors(flat_grad, grads)
            for grad, unflat_grad in zip(grads, unflat_grads):
                grad.data.copy_(unflat_grad)
        
        # Register hook on last parameter in bucket (reverse order)
        bucket_params[-1].register_post_accumulate_grad_hook(bucket_hook)

    def forward(self, *inputs, **kwargs):
        """Forward pass through wrapped module"""
        return self.module(*inputs, **kwargs)

class TensorShardedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        """
        Sharded optimizer wrapper.
        - params: model.parameters() or param groups
        - optimizer_cls: e.g., torch.optim.AdamW
        - kwargs: optimizer hyperparams
        """
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized")

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.full_params: List[torch.nn.Parameter] = []

        # Store optimizer class and kwargs for delayed creation
        self._optimizer_cls = optimizer_cls
        self._optimizer_kwargs = kwargs
        self._optim = None  # will be created on first add_param_group

        # Call base Optimizer to normalize params and call add_param_group
        super().__init__(params, kwargs)

    def add_param_group(self, param_group: dict[str, Any]):
        """
        For each full parameter p:
          - record shard range
          - register only shard view into the wrapped optimizer
        """
        new_shards = []
        for p in param_group["params"]:
            if not p.requires_grad:
                continue

            numel = p.numel()
            shard_size = (numel + self.world_size - 1) // self.world_size
            start = self.rank * shard_size
            end = min(start + shard_size, numel)
            # create shard wrapper (a Parameter, so optimizer allocates state only for this slice)
            shard_tensor = p.data.reshape(-1)[start:end].detach().clone()
            shard_param = torch.nn.Parameter(shard_tensor, requires_grad=True)
            p._shard_param = shard_param
            self.full_params.append(p)

            p._shard_info = (start, end, self.rank, shard_param)
            new_shards.append(shard_param)

        if new_shards:
            new_group = {k: v for k, v in param_group.items() if k != "params"}
            new_group["params"] = new_shards

            if self._optim is None:
                self._optim = self._optimizer_cls([new_group], **self._optimizer_kwargs)
            else:
                self._optim.add_param_group(new_group)

        self.param_groups = self._optim.param_groups if self._optim is not None else []

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        """
        Step optimizer on local shards and write updates back to original parameters.
        """
        if self._optim is None:
            raise RuntimeError("No parameters in optimizer")

        for p in self.full_params:
            shard = p._shard_param
            if p.grad is None:
                shard.grad = None
                continue

            start, end, src, shard_param = p._shard_info
            grad_slice = p.grad.view(-1)[start:end]

            if shard.grad is None:
                # make a leaf tensor for .grad
                shard.grad = grad_slice.clone()
            else:
                shard.grad.copy_(grad_slice)
        
        for param_group in self.param_groups:
            print("-------------------------------------")
            for p in param_group['params']:
                print(p.grad)
            print("-------------------------------------")

        loss = self._optim.step(closure=closure, **kwargs)

        # Copy updated shard values back into full parameters
        for p in self.full_params:
            start, end, src, shard_param = p._shard_info
            p.data.flatten()[start:end].copy_(shard_param.data)

        dist.barrier()
        self._sync_parameters()

        return loss

    def _sync_parameters(self):
        """Synchronize parameters by broadcasting each shard from its owner rank"""
        # Sort parameters by shard source to ensure consistent communication order
        for p in self.full_params:
            numel = p.numel()
            shard_size = (numel + self.world_size - 1) // self.world_size
            for i in range(self.world_size):
                start = i * shard_size
                end = min(start + shard_size, numel)
                # Flatten the full parameter tensor
                flat_param = p.data.flatten()
                # Only the owner rank broadcasts, others receive from owner
                dist.broadcast(flat_param[start: end], src=i)

    def zero_grad(self, set_to_none: bool = False):
        # zero inner optimizer (shard params)
        if self._optim is not None:
            self._optim.zero_grad(set_to_none=set_to_none)

        # also clear full-parameter grads (so next backward starts clean)
        for p in self.full_params:
            if set_to_none:
                p.grad = None
            else:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
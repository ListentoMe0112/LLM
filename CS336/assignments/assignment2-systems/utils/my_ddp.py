import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

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

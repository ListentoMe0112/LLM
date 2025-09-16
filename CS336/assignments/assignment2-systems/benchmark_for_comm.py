import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import time
from statistics import mean

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def benchmark_all_reduce(rank, world_size, args):
    # Set CUDA device for NCCL backend
    if args.device == "cuda":
        torch.cuda.set_device(rank % torch.cuda.device_count())
    
    setup(rank, world_size, args.backend)
    
    # Calculate tensor size (float32: 4 bytes per element)
    element_size = args.data_size * 1024**2 // 4
    tensor = torch.rand(element_size, dtype=torch.float32)
    
    if args.device == "cuda":
        tensor = tensor.cuda()
    
    # Warm-up
    for _ in range(3):
        dist.all_reduce(tensor, async_op=False)
    
    # Timing
    timings = []
    for _ in range(args.iterations):
        if args.device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start = time.perf_counter()
        
        dist.all_reduce(tensor, async_op=False)
        
        if args.device == "cuda":
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))
        else:
            end = time.perf_counter()
            timings.append((end - start) * 1000)  # Convert to milliseconds

    # Collect results from all processes
    avg_time = mean(timings)
    results = [torch.tensor(avg_time).to(tensor.device) for _ in range(world_size)]
    dist.all_gather(results, torch.tensor(avg_time).to(tensor.device))
    
    # Print from master process
    if rank == 0:
        max_time = max([t.item() for t in results])
        print(f"Backend: {args.backend}, Device: {args.device}, "
              f"Data Size: {args.data_size}MB, Processes: {world_size}, "
              f"Avg Time: {max_time:.3f}ms")

def main(args):
    # Validate backend-device combination
    if args.backend == "nccl" and args.device != "cuda":
        raise ValueError("NCCL backend requires CUDA device")
    
    # Limit visible GPUs
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(args.num_processes))
    
    print(f"\n=== Starting benchmark: {args.backend}-{args.device} "
          f"{args.data_size}MB {args.num_processes} processes ===")
    
    mp.spawn(fn=benchmark_all_reduce,
             args=(args.num_processes, args),
             nprocs=args.num_processes,
             join=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="All-Reduce Benchmark")
    parser.add_argument("--backend", choices=["gloo", "nccl"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--data_size", type=int, 
                       choices=[1, 10, 100, 1024], required=True)
    parser.add_argument("--num_processes", type=int, 
                       choices=[2, 4, 6], required=True)
    parser.add_argument("--iterations", type=int, default=10,
                       help="Number of measurement iterations")
    
    args = parser.parse_args()
    main(args)
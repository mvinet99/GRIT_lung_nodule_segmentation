#!/usr/bin/env python
"""10-second multi-GPU communication test (no data needed).

  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --standalone --nproc_per_node=3 nccl_check.py

If this hangs / times out, NCCL peer-to-peer is broken on this machine. Retry with:
  NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3 torchrun --standalone --nproc_per_node=3 nccl_check.py
"""
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl", timeout=timedelta(minutes=2))
rank, world = dist.get_rank(), dist.get_world_size()
print(f"rank {rank}: GPU {torch.cuda.get_device_name(local_rank)} "
      f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})", flush=True)

t0 = time.time()
x = torch.ones(1, device="cuda") * (rank + 1)
dist.all_reduce(x)                                     # small message
y = torch.randn(64 * 1024 * 1024 // 4, device="cuda")  # 64 MB, like a gradient bucket
dist.all_reduce(y)
dist.broadcast(y, src=0)                               # what DDP does at start-up
torch.cuda.synchronize()
if rank == 0:
    ok = x.item() == world * (world + 1) / 2
    print(f"NCCL {'OK' if ok else 'WRONG RESULT'}: all_reduce + broadcast across {world} GPUs "
          f"in {time.time() - t0:.2f}s", flush=True)
dist.barrier()
dist.destroy_process_group()

# test_nccl.py
import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Rank {rank}/{world_size} NCCL init success")
    dist.barrier()
    print(f"Rank {rank} barrier passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

import torch
import torch.distributed as dist
# from collections.abc import Callable

def ddp_dmo(dmo, losses, rank, world_size, src_device="cuda:0"):
    # quantilize the current micro-batch of losses
    sort, argsort = torch.sort(losses, stable=True)
    micro_size = len(sort)

    # allocate memory for full vector of losses then gather
    gather_list = [torch.zeros(micro_size).to(src_device) for r in range(world_size)] if rank == 0 else None
    dist.gather(sort, gather_list=gather_list, dst=0)

    # if source rank, then apply dmo to gathered list
    if rank == 0:
        q = dmo(torch.cat(gather_list))
        scatter_list = [q[micro_size * r: micro_size * (r + 1)].to(src_device) for r in range(world_size)]
    else:
        scatter_list = None

    q_batch = torch.zeros(micro_size).to(f"cuda:{rank}")
    dist.scatter(q_batch, scatter_list=scatter_list, src=0)

    return q_batch[torch.argsort(argsort)]
import torch
import torch.distributed as dist
import os

def ddp_max_oracle(max_oracle, losses, micro_sizes, src_device=0):
    """Take any existing maximization oracle and apply it to multiple devices using a gather-scatter implementation.
    Assumes that process rank is discoverable,~e.g. the job is run using `torchrun`.
    
    Args:
      max_oracle: a function that consumes ``n`` (full-batch size) loss values and returns ``n`` weights
      losses: a PyTorch tensor of ``micro_sizes[rank]`` losses
      micro_sizes: an array-like object of integers, indicating the micro-batch size within each process
    
    Returns:
      weights
        a vector of weights of size ``micro_sizes[rank]`` indicating the weight on each example
    """

    # type check
    assert callable(max_oracle), f"`max_oracle` must be a function, got {type(max_oracle)} instead"
    assert torch.is_tensor(losses), f"`losses` must by a PyTorch tensor, got {type(losses)} instead"

    # discover the index of this process
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert len(micro_sizes) == world_size, f"number of processes ({world_size}) should be equal to number of micro-batch sizes ({len(micro_sizes)})"
    assert micro_sizes[rank] == len(losses), f"number of losses ({len(losses)}) supplied should be equal to element {rank} of micro_sizes"

    # quantilize the current micro-batch of losses
    sort, argsort = torch.sort(losses, stable=True)

    # allocate memory for full vector of losses then gather
    gather_list = [torch.zeros(m).to(src_device) for m in micro_sizes] if rank==0 else None
  
    # print(f"RANK {rank}, SRC_DEVICE {src_device}, SORT DEVICE {sort.get_device()}")
    dist.gather(sort, gather_list=gather_list, dst=src_device)

    # if source rank, then apply dmo to gathered list
    q_batch = torch.zeros(micro_sizes[rank]).to(f"cuda:{rank}")
    if rank == 0:
        q = max_oracle(torch.cat(gather_list))
        _micro_sizes = [0]
        for i, m in enumerate(micro_sizes):
          _micro_sizes.append(_micro_sizes[i] + int(m))
        scatter_list = [q[_micro_sizes[r]: _micro_sizes[r + 1]].to(src_device) for r in range(world_size)]
        dist.scatter(q_batch, scatter_list=scatter_list, src=src_device)
    else:
        dist.scatter(q_batch, src=src_device)

    # unsort to assign weights to data indices
    return q_batch[torch.argsort(argsort)]
import os
import torch

from datetime import timedelta
from loguru import logger

# Tensor Parallelism settings
RANK = int(os.getenv("RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
PP_WORLD_SIZE = int(os.getenv("PP_WORLD_SIZE", "1"))

# CUDA memory fraction
MEMORY_FRACTION = float(os.getenv("CUDA_MEMORY_FRACTION", "1.0"))


class FakeBarrier:
    def wait(self):
        pass


class FakeGroup:
    def __init__(self, rank, size):
        self._rank = rank
        self._size = size

    def allreduce(self, *args, **kwargs):
        return FakeBarrier()

    def allgather(self, inputs, local_tensor, **kwargs):
        assert (
            len(inputs[0]) == len(local_tensor) == 1
        ), f"{len(inputs[0])} != {len(local_tensor)} != 1, and the FakeGroup is supposed to join on simple tensors"
        for input_ in inputs:
            input_[0].data = local_tensor[0].data
        return FakeBarrier()

    def barrier(self, *args, **kwargs):
        return FakeBarrier()

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def initialize_torch_distributed():
    if torch.cuda.is_available():
        from torch.distributed import ProcessGroupNCCL

        # Set the device id.
        assert WORLD_SIZE <= torch.cuda.device_count(), "Each process is one gpu"
        device = RANK % torch.cuda.device_count()
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, device)
        backend = "nccl"
        options = ProcessGroupNCCL.Options()
        options.is_high_priority_stream = True
        options._timeout = timedelta(seconds=60)
    else:
        backend = "gloo"
        options = None

    # Return: Total_Group, TP_Group, PP_Group, RANK, WORLD_SIZE, TP_WORLD_SIZE, PP_WORLD_SIZE
    if WORLD_SIZE == 1:
        assert WORLD_SIZE == PP_WORLD_SIZE, f"WORLD_SIZE(which is {WORLD_SIZE}) is not equal to PP_WORLD_SIZE(which is {PP_WORLD_SIZE})"
        TP_WORLD_SIZE = WORLD_SIZE
        return FakeGroup(RANK, WORLD_SIZE), FakeGroup(RANK, TP_WORLD_SIZE), None, RANK, WORLD_SIZE, TP_WORLD_SIZE, PP_WORLD_SIZE
    else:
        if os.getenv("DEBUG", None) == "1":
            raise NotImplementedError

        if not torch.distributed.is_initialized():
            # Call the init process.
            torch.distributed.init_process_group(
                backend=backend,
                world_size=WORLD_SIZE,
                rank=RANK,
                timeout=timedelta(seconds=60),
                pg_options=options,
            )

            if PP_WORLD_SIZE == 1: 
                TP_WORLD_SIZE = WORLD_SIZE
                return torch.distributed.group.WORLD, torch.distributed.group.WORLD, None, RANK, WORLD_SIZE, TP_WORLD_SIZE, PP_WORLD_SIZE
            else:
                assert PP_WORLD_SIZE == 2, f"Only Support PP_WORLD_SIZE == 2"
                assert WORLD_SIZE % PP_WORLD_SIZE == 0, f"WORLD_SIZE(which is {WORLD_SIZE}) is not divisible by PP_WORLD_SIZE(which is {PP_WORLD_SIZE})"
                TP_WORLD_SIZE = WORLD_SIZE // PP_WORLD_SIZE
                tp_begin_rank = RANK // TP_WORLD_SIZE * TP_WORLD_SIZE
                tp_ranks = [tp_begin_rank + i for i in range(tp_begin_rank)]
                tp_group = torch.distributed.new_group(
                    rank=tp_ranks, 
                    timeout=timedelta(seconds=60), 
                    backend=backend, 
                    pg_options=options,
                )
                pp_peer_rank = (RANK + TP_WORLD_SIZE) if (RANK < TP_WORLD_SIZE) else (RANK - TP_WORLD_SIZE)
                pp_ranks = [RANK, pp_peer_rank]
                pp_ranks.sort()
                pp_group = torch.distributed.new_group(
                    rank=pp_ranks, 
                    timeout=timedelta(seconds=60), 
                    backend=backend, 
                    pg_options=options,
                )
                return torch.distributed.group.WORLD, tp_group, pp_group, RANK, WORLD_SIZE, TP_WORLD_SIZE, PP_WORLD_SIZE
        else:
            raise "torch.distributed is already initialized."
        

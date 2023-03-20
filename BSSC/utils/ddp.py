import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class DdpSetter(object):
    def __init__(self, **kwargs):
        self.num_gpus = torch.cuda.device_count()

    def setup(self, model):
        # initialize the process group
        # dist.init_process_group("gloo", rank=self.rank, world_size=self.world_size)
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device_id = rank % torch.cuda.device_count()
        ddp_model = DDP(model, device_ids=[device_id])

        return ddp_model, rank, device_id

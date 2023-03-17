import os
import logging

import torch
import torch.multiprocessing as mp

from typing import List, Optional
from cogdl.wrappers import ModelWrapper

log = logging.getLogger(__name__)


class TrainingController(object):
    def __init__(self, device_ids: Optional[List[int]], dist: str = "ddp", backend: str = "nccl"):
        self.device_ids = device_ids
        self.backend = backend

    def init_controller(self):
        if self.backend == "ddp":
            pass
        elif self.backend == "dp":
            pass
        else:
            raise NotImplementedError

    def setup(self, model_w: ModelWrapper):
        mp.spawn(self.new_process, args=(), nprocs=2)

    def new_process(self) -> None:
        pass

    def init_ddp(self, global_rank: Optional[int], world_size: Optional[int]) -> None:
        # TODO: this code is duplicated in DDP and DDPSpawn, make this a function
        global_rank = global_rank if global_rank is not None else self.global_rank()
        world_size = world_size if world_size is not None else self.world_size()
        os.environ["MASTER_ADDR"] = self.master_address()
        os.environ["MASTER_PORT"] = str(self.master_port())

        if not torch.distributed.is_initialized():
            log.info(f"initializing ddp: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}")
            torch.distributed.init_process_group(self.backend, rank=global_rank, world_size=world_size)

    @property
    def torch_distributed_backend(self):
        torch_backend = os.getenv("PL_TORCH_DISTRIBUTED_BACKEND")
        if torch_backend is None:
            torch_backend = "nccl" if self.on_gpu else "gloo"
        return torch_backend

    def local_rank(self) -> int:
        return int(os.environ["LOCAL_RANK"])

    def master_address(self) -> str:
        if "MASTER_ADDR" not in os.environ:
            # rank_zero_warn("MASTER_ADDR environment variable is not defined. Set as localhost")
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        log.debug(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        master_address = os.environ.get("MASTER_ADDR")
        return master_address

    def master_port(self) -> int:
        if "MASTER_PORT" not in os.environ:
            # rank_zero_warn("MASTER_PORT environment variable is not defined. Set as 12910")
            os.environ["MASTER_PORT"] = "12910"
        log.debug(f"MASTER_PORT: {os.environ['MASTER_PORT']}")

        port = int(os.environ.get("MASTER_PORT"))
        return port

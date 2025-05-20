import os
import wandb
import torch
import torch.distributed as dist
import numpy as np
import random
from lib.loggers import WandbLogger, NoLogger
import torch.nn as nn
from datetime import timedelta

class AcumenAccelerator(object):
    _instance = None

    def __initialize(self):
        self.args = None

        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self._is_distributed = int(os.environ.get('RANK', -1)) != -1

        print(f"[💥 Accelerator 💥] Rank: {self.rank}, Local Rank: {self.local_rank}, World Size: {self.world_size}")

        if self._is_distributed:
            dist.init_process_group(
                backend = 'nccl',
                timeout = timedelta(seconds = 7200000)
            )
            device = f'cuda:{self.local_rank}'
            torch.cuda.set_device(device)

        self.master_print('💥💥💥 Acumen Accelerator Initialized 💥💥💥')

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AcumenAccelerator, cls).__new__(cls)
            cls._instance.__initialize()
        return cls._instance

    def synchronize(self):
        if self.is_distributed:
            torch.cuda.synchronize()

    def get_logger(self):
        if self.rank == 0:
            return WandbLogger()

        return NoLogger()

    def master_print(self, *args, **kwargs):
        if self.rank == 0:
            print(*args, **kwargs)

    def is_master(self):
        return self.rank == 0

    def prepare_model(self, model):
        from lib import device

        if self.args.use_compile:
            model = torch.compile(model)
            self.master_print(f"💥 Model compiled successfully")
        else:
            self.master_print(f"💥 Not compiling (use_compile = False).")

        if self.is_distributed:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = model.to(device)

            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids = [self.local_rank],
                output_device = self.local_rank,
            )
            print(f"💥 Using DDP with local rank {self.local_rank} ...")

        elif not hasattr(model, 'module'):
            # model = nn.DataParallel(model)
            model = model.to(device)
            print(f"💥 Using DataParallel ...")

        return model

    def set_seed(self):
        if not ('seed' in self.args and self.args.seed != -1):
            return

        torch.manual_seed(self.args.seed + self.rank)
        torch.cuda.manual_seed(self.args.seed + self.rank)
        np.random.seed(self.args.seed + self.rank)
        random.seed(self.args.seed + self.rank)

        print(f'[💥 Rank - {self.local_rank}] Setting random seed to {self.args.seed + self.rank} 🌱🌱🌱')

    def set_rng_state(self, state):
        torch.set_rng_state(torch.tensor(state['torch'][self.local_rank], dtype = torch.uint8))
        torch.cuda.set_rng_state(torch.tensor(state['cuda'][self.local_rank], dtype = torch.uint8))
        np.random.set_state(state['numpy'][self.local_rank])
        random.setstate(state['random'][self.local_rank])

    def gather_rng_states(self):
        state = {
            'torch': torch.get_rng_state().numpy(),
            'cuda': torch.cuda.get_rng_state().numpy(),
            'numpy': np.random.get_state(),
            'random': random.getstate(),
        }

        if not self.is_distributed:
            return {k: [v] for k,v in state.items()}

        states = [None for _ in range(self.world_size)]
        dist.all_gather_object(states, state['torch'])

        state['torch'] = states

        states = [None for _ in range(self.world_size)]
        dist.all_gather_object(states, state['cuda'])

        state['cuda'] = states

        states = [None for _ in range(self.world_size)]
        dist.all_gather_object(states, state['numpy'])

        state['numpy'] = states

        states = [None for _ in range(self.world_size)]
        dist.all_gather_object(states, state['random'])

        state['random'] = states

        return state

    def set_args(self, args):
        self.args = args
        self.set_seed()

    def terminate(self):
        if self._is_distributed:
            dist.destroy_process_group()

    @property
    def is_distributed(self):
        return self._is_distributed

    def prepare_loggers(self):
        if not self.is_master():
            return

        os.environ['WANDB_MODE'] = self.args.mode
        os.environ['WANDB_NAME'] = self.args.name
        os.environ['WANDB_NOTES'] = self.args.notes

        if self.args.resume_from != '':
            wandb.init(
                project = 'san',
                group = self.args.group,
                entity = self.args.environment.extra_args.wandb_entity,
                id = self.args.wandb_run_id,
                resume = 'must'
            )
        else:
            self.args.wandb_run_id = wandb.util.generate_id()

            wandb.init(
                project = 'san',
                group = self.args.group,
                entity = self.args.environment.extra_args.wandb_entity,
                id = self.args.wandb_run_id,
            )

        wandb.config.update(vars(self.args), allow_val_change = True)

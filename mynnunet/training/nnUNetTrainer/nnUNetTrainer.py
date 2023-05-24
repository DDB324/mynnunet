import inspect

import torch
from torch import distributed as dist
from torch.cuda import device_count


class MYnnUNetTrainer(object):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        self.is_ddp = dist.is_available() and dist.is_initialized()
        self.local_rank = 0 if not self.is_ddp else dist.get_rank()
        self.device = device

        # print what device we are using
        if self.is_ddp:
            print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. "
                  f"The world size is {dist.get_world_size()}. "
                  f"Setting device to {self.device}")
            self.device = torch.device(type='cuda', index=self.local_rank)
        else:
            if self.device.type == 'cuda':
                self.device = torch.device(type='cuda', index=0)
            print(f"Using device: {self.device}")

        # loading and saving this class for continuing from checkpoint should not happen based on pickling.
        # This would also pickle the network etc.
        # Instead, we just reinstantiate and then load the checkpoint we need,
        # So let's save the init args
        # self.my_init_kwargs = {}
        # for k in inspect.signature(self.__init__).parameters.keys():
        #     self.my_init_kwargs[k] = locals()[k]
        self.my_init_kwargs = {k: v for k, v in locals().items() if k in inspect.signature(self.__init__).parameters}


def my_function(x, y):
    z = x + y
    a = locals().items()
    for k, v in a:
        print(k, v)


if __name__ == '__main__':
    my_function(2, 3)

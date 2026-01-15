# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class WarmUp(LambdaLR):
    def __init__(self, optimizer: Optimizer, n_steps: int = 1000):
        super().__init__(optimizer, lambda epoch: min((epoch - 1) / (n_steps - 1), 1.0))


class WarmUpStepLR(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 1000,
        decay_start_step: int = 50000,
        decay_stop_step: int = 50000,
        step_size: int = 1500,
        decay: float = 0.98,
    ):
        super().__init__(
            optimizer,
            lambda epoch: (
                min((epoch - 1) / (warmup_steps - 1), 1.0)
                if epoch < decay_start_step
                else decay ** ((min(epoch, decay_stop_step) - decay_start_step) // step_size)
            ),
        )

import torch
from torch.optim.lr_scheduler import (
    _LRScheduler,
    CosineAnnealingLR as TorchCosineAnnealingLR,
)


class LinearWarmupLR(_LRScheduler):
    """Decorator of learning rate scheduler which applies linear warmup."""

    def __init__(
        self,
        scheduler: _LRScheduler,
        warmup_steps: int,
        init_lr_ratio: float,
    ):
        self.scheduler = scheduler
        self.steps = warmup_steps
        self.init_ratio = init_lr_ratio

        super().__init__(scheduler.optimizer)

    def step(self):
        step = self.last_epoch + 1
        if (step and not self.steps) or step > self.steps:
            self.scheduler.step()

        super().step()

    def get_lr(self):
        step = self.last_epoch
        if not self.steps or step > self.steps:
            return self.scheduler.get_last_lr()

        factor = 1 - (1 - step / self.steps) * (1 - self.init_ratio)
        return [pg['initial_lr'] * factor for pg in self.optimizer.param_groups]


class CosineAnnealingLR(TorchCosineAnnealingLR):
    """Learning rate scheduler following cosine wave down to minimum factor."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        min_lr_ratio: float,
    ):
        lrs = torch.tensor([pg['lr'] for pg in optimizer.param_groups])
        self.min_lrs = lrs * min_lr_ratio

        super().__init__(optimizer, T_max)

    def get_lr(self):        
        lrs = super().get_lr()
        return [max(min_lr, lr) for min_lr, lr in zip(self.min_lrs, lrs)]

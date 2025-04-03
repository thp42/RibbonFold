import os, sys
import torch
from torch.optim.lr_scheduler import _LRScheduler

class WarmLR(_LRScheduler):
    """Add a small value (pos or neg) to the current lr util the given bound
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_theta: float or list
        final_lr: float or list
            The max or min lr
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """
    
    def __init__(self, optimizer, lr_theta, final_lr, last_epoch=-1, verbose=False):
        self.optimizer = optimizer
        assert type(lr_theta) == type(final_lr), "Same type of lr_theta and final_lr is required"
        assert isinstance(lr_theta, float) or isinstance(lr_theta, list) or isinstance(lr_theta, tuple)

        if not isinstance(lr_theta, list) and not isinstance(lr_theta, tuple):
            self.lr_thetas = [lr_theta] * len(optimizer.param_groups)
            self.final_lrs = [final_lr] * len(optimizer.param_groups)
        else:
            if len(lr_theta) != len(optimizer.param_groups) or len(final_lr) != len(optimizer.param_groups):
                raise ValueError("Expected {} lr_thetas and final_lr, but got {}, {}".format(
                    len(optimizer.param_groups), len(lr_theta), len(final_lr)))
            self.lr_thetas = list(lr_theta)
            self.final_lrs = list(final_lr)
        super(WarmLR, self).__init__(optimizer, last_epoch, verbose)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.
        """
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_thetas', 'final_lrs')}
        state_dict['lr_thetas'] = self.lr_thetas[:]
        state_dict['final_lrs'] = self.final_lrs[:]

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        lr_thetas = state_dict.pop('lr_thetas')
        final_lrs = state_dict.pop('final_lrs')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['lr_thetas'] = lr_thetas
        state_dict['final_lrs'] = final_lrs




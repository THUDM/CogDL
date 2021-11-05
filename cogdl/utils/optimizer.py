"""A wrapper class for optimizer """
import numpy as np
import torch.nn as nn


class NoamOptimizer(nn.Module):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, d_model, n_warmup_steps, init_lr=None):
        super(NoamOptimizer, self).__init__()
        self._optimizer = optimizer
        self.param_groups = optimizer.param_groups
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5) if init_lr is None else init_lr / np.power(self.n_warmup_steps, -0.5)

    def step(self):
        """Step with the inner optimizer"""
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer"""
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min(
            [np.power(self.n_current_steps, -0.5), np.power(self.n_warmup_steps, -1.5) * self.n_current_steps]
        )

    def _update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr


class LinearOptimizer(nn.Module):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, n_warmup_steps, n_training_steps, init_lr=0.001):
        super(LinearOptimizer, self).__init__()
        self._optimizer = optimizer
        self.param_groups = optimizer.param_groups
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.n_training_steps = n_training_steps
        self.init_lr = init_lr

    def step(self):
        """Step with the inner optimizer"""
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer"""
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        if self.n_current_steps < self.n_warmup_steps:
            return float(self.n_current_steps) / float(max(1, self.n_warmup_steps))
        return max(
            0.0,
            float(self.n_training_steps - self.n_current_steps)
            / float(max(1, self.n_training_steps - self.n_warmup_steps)),
        )

    def _update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

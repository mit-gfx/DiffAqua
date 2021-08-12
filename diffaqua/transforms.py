from typing import Sequence, Optional
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CreateChannel(object):
    """Create the channel dimension in front of all other dimensions.

    All other operations should be done after the channel is created.
    """

    def __call__(self, tensor: Tensor):
        return tensor.unsqueeze(0)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Inverse(object):
    def __call__(self, tensor: Tensor):
        tensor = 1 - tensor
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '()'


class PadToSize(object):
    def __init__(
            self,
            target_sizes: Sequence[int],
            value: float = 0.0,
            center: Optional[Sequence[Optional[int]]] = None
        ):

        self.target_sizes = list(target_sizes)
        self.value = value
        self.center = [None] * len(target_sizes) if center is None else center

        if len(self.center) != len(target_sizes):
            raise ValueError(
                "Target {0} mismatch with center {1} in dimension".format(
                    len(target_sizes), len(self.center)))

    def __call__(self, tensor: Tensor):
        target_sizes = self.target_sizes
        center = self.center

        if tensor.dim() - 1 != len(target_sizes):
            raise ValueError(
                "Source dimension {0} mismatch with target dimension {1}".format(
                    tensor.dim() -1, len(target_sizes)))

        source_sizes = tensor.size()[1:]
        for s_size, t_size in zip(source_sizes, target_sizes):
            if s_size > t_size:
                raise ValueError(
                    "Source size {0} is larger than target size {1}".format(
                        source_sizes, target_sizes))

        pad = []
        for s_size, t_size, center in zip(source_sizes, target_sizes, self.center):
            double_left_offset = s_size if center is None else 2 * center
            pad_left = math.floor((t_size - double_left_offset) / 2)
            pad_right = t_size - s_size - pad_left
            pad.append([pad_left, pad_right])
        pad = [item for subpad in reversed(pad) for item in subpad]

        tensor = F.pad(tensor, pad, mode='constant', value=self.value)

        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(target_sizes={0}, value={1}, center={2})'.format(
            self.target_sizes, self.value, self.center)


class ProbNormalize(object):
    def __init__(self, norm_type: str = 'sum'):
        self.norm_type = norm_type

        if norm_type == 'sum':
            self.reduce_func = torch.sum
        elif norm_type == 'mean':
            self.reduce_func = torch.mean
        elif norm_type == 'max':
            self.reduce_func = torch.max
        else:
            raise ValueError(
                "Normalization type must be 'sum', 'mean' or 'max', not {0}".format(norm_type))

    def __call__(self, tensor: Tensor):
        reduce_func = self.reduce_func

        bias = tensor.new_zeros(tensor.size(0))
        scale = tensor.new_ones(tensor.size(0))

        for i in range(tensor.size(0)):
            bias[i] = tensor[i].min()
        bias = bias.view([-1] + [1] * (tensor.dim() - 1))
        tensor = tensor - bias

        for i in range(tensor.size(0)):
            scale[i] = 1 / reduce_func(tensor[i])

        scale = scale.view([-1] + [1] * (tensor.dim() - 1))
        tensor = tensor * scale

        tensor[torch.isnan(tensor)] = 0.0

        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(norm_type={0})'.format(self.norm_type)

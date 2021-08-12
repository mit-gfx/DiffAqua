import os
import sys
import math
import time
from pathlib import Path
from collections import OrderedDict

from tqdm import tqdm, trange
import numpy as np
import scipy.optimize
from scipy.optimize import fsolve

import torch
import torch.nn as nn

from .nn import GaussianConv3d


class WassersteinInterpolate(nn.Module):
    def __init__(
            self, bases, muscles, kernel_size=None, sigma=1.0,
            niter=10, tol=1e-7, eps=1e-7, entropy_lmt=None, norm_type='max'):

        super().__init__()

        bases_size = list(bases.size())
        self.register_buffer('bases', bases, persistent=False)
        self.register_buffer('muscles', muscles, persistent=False)
        self.register_buffer('area_weights',
            torch.ones([1, 1] + bases_size[2:]).div(np.prod(bases_size[2:])),
            persistent=False)
        self.register_buffer('inv_variances',
            1 / bases.var(dim=list(range(1, bases.dim()))), persistent=False)

        if kernel_size is None:
            kernel_size = max(bases_size[2:])
        dtype = bases.dtype
        self.gaussian_conv = GaussianConv3d(sigma, kernel_size, pad_mode='zero', dtype=dtype)

        self.niter = niter
        self.tol = tol
        self.eps = eps
        self.entropy_lmt = entropy_lmt

        if norm_type not in ['max', 'var']:
            raise ValueError("Norm type should be one of ['max', 'var'], got {}".format(norm_type))
        self.norm_type = norm_type

        self.alpha = nn.Parameter(torch.ones(bases_size[0]).div(bases_size[0]))

    @torch.no_grad()
    def alpha_one_hot_(self, index):
        self.alpha.data.zero_()
        self.alpha.data[index] = 1.0

    def forward(self):

        gaussian_conv = self.gaussian_conv
        gaussian_conv_t = gaussian_conv
        area_weights = self.area_weights
        inv_variances = self.inv_variances

        niter = self.niter
        tol = self.tol
        eps = self.eps
        entropy_lmt = self.entropy_lmt

        bases = self.bases
        muscles = self.muscles

        w = torch.ones_like(bases)
        v = torch.ones_like(bases)
        barycenter = torch.ones_like(bases[0]).unsqueeze(0)

        alpha = self.alpha
        alpha = alpha.view([-1] + [1] * (bases.dim() - 1))

        for i in range(niter):
            prev_barycenter = barycenter.clone()
            prev_v = v.clone()

            w = bases / (gaussian_conv_t(v * area_weights) + eps)

            d = v * gaussian_conv(w * area_weights)

            barycenter = (d.add(eps).log() * alpha).sum(dim=0, keepdim=True).exp()

            if entropy_lmt is not None:
                entropy = -(barycenter.log() * barycenter * area_weights).sum()
                if i > 0 and entropy > entropy_lmt:
                    result = fsolve(
                        WassersteinInterpolate.entropy_func,
                        args=(barycenter, area_weights, entropy_lmt),
                        x0=1.0, epsfcn=1e-2, xtol=1e-4, full_output=True)
                    if result[2] == 1:
                        a = result[0][0]
                    else:
                        a = 1.0
                        print('projection failed')
                    barycenter = barycenter**a

            v = v * barycenter / (d + eps)

            if torch.isnan(barycenter).any() or torch.isinf(barycenter).any():
                barycenter = prev_barycenter
                v = prev_v
                break

            with torch.no_grad():
                change = ((prev_barycenter - barycenter).abs() * area_weights).sum().item()

            if i > 0 and change < tol:
                break

        if self.norm_type == 'var':
            barycenter = barycenter * inv_variances.dot(alpha.view(-1))
        elif self.norm_type == 'max':
            barycenter = barycenter / barycenter.max().item()

        alpha = alpha.view([-1] + [1] * (muscles.dim() - 1))
        muscles = (alpha * muscles).sum(0)

        return barycenter, v, muscles

    @staticmethod
    def rpy_to_rotation_matrix(euler_angles):
        roll, pitch, yaw = euler_angles
        c_r, s_r = torch.cos(roll), torch.sin(roll)
        c_p, s_p = torch.cos(pitch), torch.sin(pitch)
        c_y, s_y = torch.cos(yaw), torch.sin(yaw)

        R_roll = torch.eye(3).to(euler_angles.device)
        R_pitch = torch.eye(3).to(euler_angles.device)
        R_yaw = torch.eye(3).to(euler_angles.device)

        R_roll[1, 1] = c_r
        R_roll[1, 2] = -s_r
        R_roll[2, 1] = s_r
        R_roll[2, 2] = c_r

        R_pitch[0, 0] = c_p
        R_pitch[0, 2] = s_p
        R_pitch[2, 0] = -s_p
        R_pitch[2, 2] = c_p

        R_yaw[0, 0] = c_y
        R_yaw[0, 1] = -s_y
        R_yaw[1, 0] = s_y
        R_yaw[1, 1] = c_y

        return torch.mm(R_yaw, torch.mm(R_pitch, R_roll))

    def muscle_density(self, barycenter, muscles):
        if (muscles[2] == 0.0).all():
            xyz = np.stack(np.meshgrid(*[np.arange(t) for t in barycenter.size()[-3:]], indexing='ij'), axis=-1)
            xyz = torch.Tensor(xyz).to(self.alpha.device)
            xyz = xyz + 0.5
            xyz = xyz - muscles[1]
            xyz = xyz / muscles[0]
            xyz = torch.square(xyz)
            xyz = -xyz
            xyz = torch.exp(xyz)
            density = torch.mean(xyz, dim=-1)
            xyz_mask = (torch.sum((xyz.cpu() > 0.5), dim=-1) == 3).double().to(self.alpha.device).detach()
            density = xyz_mask * density
        else:
            R = WassersteinInterpolate.rpy_to_rotation_matrix(muscles[2])
            sigma = torch.diag(muscles[0])
            sigma = R @ sigma @ R.t()
            xyz = np.stack(np.meshgrid(*[np.arange(t) for t in barycenter.size()[-3:]], indexing='ij'), axis=-1)
            xyz = torch.Tensor(xyz).to(self.alpha.device)
            xyz = xyz + 0.5
            xyz = xyz - muscles[1]
            xyz_sigma = (xyz.view(-1, 3) @ sigma.inverse()).view_as(xyz)
            xyz = torch.sum(xyz_sigma * xyz, dim=-1)
            xyz = -xyz
            density = torch.exp(xyz)
            xyz_mask = (density.cpu() > 0.5).double().to(self.alpha.device).detach()
            density = xyz_mask * density
        return density

    @staticmethod
    def entropy_func(x, barycenter, area_weights, entropy_lmt):
        x = barycenter.new_tensor(x)
        ret = -(area_weights * x * barycenter**x * barycenter.log()).sum() - entropy_lmt
        return ret.item()


class LinearInterpolate(nn.Module):
    def __init__(
            self, bases, muscles):

        super().__init__()

        bases_size = list(bases.size())
        self.register_buffer('bases', bases, persistent=False)
        self.register_buffer('muscles', muscles, persistent=False)

        self.alpha = nn.Parameter(torch.ones(bases_size[0]).div(bases_size[0]))

    @torch.no_grad()
    def alpha_one_hot_(self, index):
        self.alpha.data.zero_()
        self.alpha.data[index] = 1.0

    def forward(self):

        bases = self.bases
        muscles = self.muscles

        w = torch.ones_like(bases)
        v = torch.ones_like(bases)
        barycenter = torch.ones_like(bases[0]).unsqueeze(0)

        alpha = self.alpha
        alpha = alpha.view([-1] + [1] * (bases.dim() - 1))
        barycenter = (alpha * self.bases).sum(0)

        alpha = alpha.view([-1] + [1] * (muscles.dim() - 1))
        muscles = (alpha * self.muscles).sum(0)

        return barycenter, v, muscles

    @staticmethod
    def rpy_to_rotation_matrix(euler_angles):
        roll, pitch, yaw = euler_angles
        c_r, s_r = torch.cos(roll), torch.sin(roll)
        c_p, s_p = torch.cos(pitch), torch.sin(pitch)
        c_y, s_y = torch.cos(yaw), torch.sin(yaw)

        R_roll = torch.eye(3).to(euler_angles.device)
        R_pitch = torch.eye(3).to(euler_angles.device)
        R_yaw = torch.eye(3).to(euler_angles.device)

        R_roll[1, 1] = c_r
        R_roll[1, 2] = -s_r
        R_roll[2, 1] = s_r
        R_roll[2, 2] = c_r

        R_pitch[0, 0] = c_p
        R_pitch[0, 2] = s_p
        R_pitch[2, 0] = -s_p
        R_pitch[2, 2] = c_p

        R_yaw[0, 0] = c_y
        R_yaw[0, 1] = -s_y
        R_yaw[1, 0] = s_y
        R_yaw[1, 1] = c_y

        return torch.mm(R_yaw, torch.mm(R_pitch, R_roll))

    def muscle_density(self, barycenter, muscles):
        if (muscles[2] == 0.0).all():
            xyz = np.stack(np.meshgrid(*[np.arange(t) for t in barycenter.size()[-3:]], indexing='ij'), axis=-1)
            xyz = torch.Tensor(xyz).to(self.alpha.device)
            xyz = xyz + 0.5
            xyz = xyz - muscles[1]
            xyz = xyz / muscles[0]
            xyz = torch.square(xyz)
            xyz = -xyz
            xyz = torch.exp(xyz)
            density = torch.mean(xyz, dim=-1)
            xyz_mask = (torch.sum((xyz.cpu() > 0.5), dim=-1) == 3).double().to(self.alpha.device).detach()
            density = xyz_mask * density
        else:
            R = WassersteinInterpolate.rpy_to_rotation_matrix(muscles[2])
            sigma = torch.diag(muscles[0])
            sigma = R @ sigma @ R.t()
            xyz = np.stack(np.meshgrid(*[np.arange(t) for t in barycenter.size()[-3:]], indexing='ij'), axis=-1)
            xyz = torch.Tensor(xyz).to(self.alpha.device)
            xyz = xyz + 0.5
            xyz = xyz - muscles[1]
            xyz_sigma = (xyz.view(-1, 3) @ sigma.inverse()).view_as(xyz)
            xyz = torch.sum(xyz_sigma * xyz, dim=-1)
            xyz = -xyz
            density = torch.exp(xyz)
            xyz_mask = (density.cpu() > 0.5).double().to(self.alpha.device).detach()
            density = xyz_mask * density
        return density


def transport(
        bases, v, alpha, area_weights, gaussian_conv,
        gaussian_conv_t=None, w=None, eps=1e-16):

    if gaussian_conv_t is None:
        gaussian_conv_t = gaussian_conv

    if w is None:
        w = bases / (gaussian_conv_t(v * area_weights) + eps)

    alpha = (alpha / alpha.sum()).view([-1] + [1] * (bases.dim() - 1))

    d = v * gaussian_conv(w * area_weights)
    barycenter = (d.add(eps).log() * alpha).sum(dim=0, keepdim=True).exp()

    return barycenter

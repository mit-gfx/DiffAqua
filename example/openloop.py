import os
from pathlib import Path
import sys
import time
import math
import random
import copy
from collections import deque
from tqdm import trange

import scipy
import scipy.optimize
import numpy as np
import trimesh

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms as tv_transforms
from torch.utils.tensorboard import SummaryWriter

from diffaqua.interp import WassersteinInterpolate
from diffaqua.cooptim import Cooptimizer
from diffaqua.nn import GaussianConv3d, VoxelToElementSchlick
from diffaqua.util import evenly_dist_weights
from diffaqua import transforms as fish_transforms
from diffaqua.models import get_model

from diffpd.fem import DeformableHex, HydrodynamicsStateForceHex
from diffpd.sim import Sim
from diffpd.nn import OpenFoldController, OpenUnfoldController
from diffpd import transforms as pd_transforms
from diffpd.mesh import MeshHex

try:
    from pyvista.utilities import xvfb
    xvfb.start_xvfb()
except OSError:
    print('not using server')


class SubCooptimizer(Cooptimizer):

    @torch.no_grad()
    def precompute(self, barycenter, muscles, path):
        path.mkdir(parents=True, exist_ok=True)
        barycenter = barycenter[0, 0].clamp(0, 1)
        np.savez_compressed(path / 'barycenter.npz', barycenter=barycenter.numpy())

        barycenter = barycenter.round()
        muscles = [m.round() * barycenter for m in muscles]
        X, Y, Z = barycenter.size()
        self.cell_size = (X, Y, Z)
        self.node_size = (X + 1, Y + 1, Z + 1)
        rest_mesh_blob = MeshHex.load(torch.ones_like(barycenter).detach().cpu().numpy(), dx=self.dx)
        rest_mesh_fish = MeshHex.load(barycenter.detach().cpu().numpy(), dx=self.dx)

        fish_to_blob_cell = dict()
        blob_to_fish_cell = dict()
        for fish_cell_idx, blob_cell_idx in zip(
                np.nditer(rest_mesh_fish.cell_indices), np.nditer(rest_mesh_blob.cell_indices)):
            fish_cell_idx, blob_cell_idx = int(fish_cell_idx), int(blob_cell_idx)
            if fish_cell_idx >= 0:
                fish_to_blob_cell[fish_cell_idx] = blob_cell_idx
                blob_to_fish_cell[blob_cell_idx] = fish_cell_idx
        fish_to_blob_node = dict()
        blob_to_fish_node = dict()
        for fish_node_idx, blob_node_idx in zip(
                np.nditer(rest_mesh_fish.node_indices), np.nditer(rest_mesh_blob.node_indices)):
            fish_node_idx, blob_node_idx = int(fish_node_idx), int(blob_node_idx)
            if fish_node_idx >= 0:
                fish_to_blob_node[fish_node_idx] = blob_node_idx
                blob_to_fish_node[blob_node_idx] = fish_node_idx

        x_mid = math.floor(X / 2) - 1
        y_mid = math.floor(Y / 2) - 1
        z_mid = math.floor(Z / 2) - 1

        x_mid_v = x_mid + 1
        y_mid_v = y_mid + 1
        z_mid_v = z_mid + 1

        spine = [v for v in rest_mesh_blob.node_indices[:, y_mid_v, z_mid_v].ravel() if v in blob_to_fish_node]
        self.spine = spine
        transform = []

        rho = 1e3
        v_water = np.array(self.v_water, dtype=np.float64)
        fish_boundary = rest_mesh_fish.boundary
        blob_boundary = np.zeros_like(fish_boundary)
        for i, node_idx in np.ndenumerate(fish_boundary):
            blob_boundary[i] = fish_to_blob_node[node_idx]

        transform.append(pd_transforms.AddHydrodynamics(
            rho,
            v_water,
            self.Cd_points,
            self.Ct_points,
            blob_boundary.T))
        self.hydro = HydrodynamicsStateForceHex(
            rho, v_water, self.Cd_points, self.Ct_points, fish_boundary.T)
        self.hydro.set_spine(
            head=blob_to_fish_node[spine[0]],
            tail=blob_to_fish_node[spine[1]])

        muscle_stiffness = self.muscle_stiffness
        all_muscles = []

        tail_muscle = muscles[0] * barycenter
        x_cross = tail_muscle.sum(0)
        y_cross = tail_muscle.sum(1)

        shared_muscles = []
        for y_half in range(y_mid_v):
            for z in range(Z):
                if x_cross[y_half, z] == 0:
                    continue
                muscle_pair = []
                for y in [y_half, Y - 1 - y_half]:
                    indices = []
                    for x in range(X):
                        if y_cross[x, z] == 0:
                            continue
                        blob_idx = int(rest_mesh_blob.cell_indices[x, y, z])
                        indices.append(blob_idx)
                    transform.append(pd_transforms.AddActuationEnergy(muscle_stiffness, [1.0, 0.0, 0.0], indices))
                    muscle_pair.append(indices)
                shared_muscles.append(muscle_pair)
        all_muscles.append(shared_muscles)

        transform = pd_transforms.Compose(transform)

        self.center = rest_mesh_blob.node_indices[x_mid_v, y_mid_v, z_mid_v]
        self.mid_line = [v for v in rest_mesh_blob.node_indices[:, y_mid_v, z_mid_v].ravel() if v in blob_to_fish_node and v != self.center]
        self.mid_line_start = self.mid_line[0]
        self.mid_line_end = self.mid_line[-1]

        q0 = torch.as_tensor(rest_mesh_blob.vertices).clone()
        q0_center = q0.view(-1, 3)[self.center]
        self.q0 = q0.view(-1, 3).sub(q0_center).view(-1)

        self.v0 = torch.zeros_like(self.q0)

        deformable = DeformableHex(
            rest_mesh_blob,
            density=self.average_density,
            dt=self.dt,
            method=self.method,
            options=self.options,
            backward_options=self.backward_options)
        deformable = transform(deformable)

        sim = Sim(deformable)

        self.rest_mesh_blob = rest_mesh_blob
        self.rest_mesh_fish = rest_mesh_fish
        self.deformable = deformable
        self.sim = sim

        self.rest_mesh_fish_elements = torch.Tensor(rest_mesh_fish.elements)

        self.fish_to_blob_cell = fish_to_blob_cell
        self.blob_to_fish_cell = blob_to_fish_cell
        self.fish_to_blob_node = fish_to_blob_node
        self.blob_to_fish_node = blob_to_fish_node

        self.controller.update_deformable(deformable)
        self.controller.update_all_muscles(all_muscles)

        torch.save(
            {
                'all_muscle': all_muscles,
                'fish_to_blob_cell': fish_to_blob_cell,
                'blob_to_fish_cell': blob_to_fish_cell,
                'fish_to_blob_node': fish_to_blob_node,
                'blob_to_fish_node': blob_to_fish_node,
            }, path / 'muscle.pth')


@torch.no_grad()
def get_bases(width, height, length):
    models = [
        get_model('lamprey', width=width, height=height, length=length),
        get_model('lamprey', width=height, height=width, length=length)]
    bases = [m['basis'] for m in models]
    bases = torch.stack(bases, dim=0)
    muscles = [m['muscle'] for m in models]
    muscles = torch.stack(muscles, dim=0)
    return bases, muscles


def main(train_controller, train_shape, joint=False):
    root = Path(__file__).resolve().parent

    if joint:
        train_controller = True
        train_shape = True
        exp_name = 'cooptim_joint'
    else:
        if train_controller and train_shape:
            exp_name = 'cooptim_alt'
        elif train_controller:
            exp_name = 'controller'
        elif train_shape:
            exp_name = 'shape'
        else:
            raise ValueError('What?')

    save_root = root / 'experiments' / 'openloop' / f'{exp_name}'
    image_path = save_root / 'images'
    ckpt_path = save_root / 'ckpt'
    save_root.mkdir(parents=True, exist_ok=True)
    image_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)

    dtype = torch.float64
    torch.set_default_dtype(dtype)

    width = 2
    height = 8
    length = 40
    kernel_size = length
    dx = 1 / length
    eps = 1e-7
    sigma = 1.0
    niter = 20

    segment_len = 2
    sin_period = 12

    shape_lr=0.05
    controller_lr=0.05
    controller_wd=0.0
    muscle_stiffness=2e6
    youngs_modulus=5e6
    poissons_ratio=0.45
    average_density=1e3
    w_sideward=10.0
    dt=3.33e-2
    num_frames=100
    target_dir = torch.Tensor([-1, 0, 0]).detach()

    num_epochs = 20

    method = 'pd_eigen'
    options = {
        'max_pd_iter': 1000, 'max_ls_iter': 10, 'abs_tol': 1e-1, 'rel_tol': 1e-1,
        'verbose': 0, 'thread_ct': 8, 'use_bfgs': 1, 'bfgs_history_size': 10, 'bfgs_fallback': 1
    }
    backward_options = {
        'max_pd_iter': 1000, 'max_ls_iter': 10, 'abs_tol': 1e-2, 'rel_tol': 1e-3,
        'verbose': 0, 'thread_ct': 8, 'use_bfgs': 1, 'bfgs_history_size': 10, 'bfgs_fallback': 1, 'fallback': 0
    }

    bases, muscles = get_bases(width, height, length)

    shape = WassersteinInterpolate(bases, muscles, kernel_size, sigma, niter, eps, eps, None, 'var')

    controller = OpenFoldController(
        None, [[[], []]],
        num_steps=num_frames, segment_len=segment_len, init_period=sin_period)

    if not train_controller:
        controller.freeze_()

    tensorboard = SummaryWriter(save_root, purge_step=1)

    cooptimizer = Cooptimizer(
        shape=shape,
        controller=controller,
        target_dir=target_dir,
        method=method,
        options=options,
        backward_options=backward_options,
        dtype=dtype,
        num_epochs=num_epochs,
        names=['Vertical', 'Horizontal'],
        shape_lr=shape_lr,
        controller_lr=controller_lr,
        controller_wd=controller_wd,
        dx=dx,
        muscle_stiffness=muscle_stiffness,
        youngs_modulus=youngs_modulus,
        poissons_ratio=poissons_ratio,
        average_density=average_density,
        w_sideward=w_sideward,
        dt=dt,
        num_frames=num_frames,
        tensorboard=tensorboard,
    )

    if joint:
        for epoch in range(num_epochs):
            cooptimizer.train_joint(image_path, ckpt_path, openloop_flag=True)
            print(controller.state_dict())
    else:
        for epoch in range(num_epochs):
            if train_controller:
                cooptimizer.train_controller(image_path, ckpt_path, openloop_flag=True)
            if train_shape:
                cooptimizer.train_shape(image_path, ckpt_path, openloop_flag=True)


if __name__ == "__main__":
    main(True, False, False)
    main(False, True, False)
    main(True, True, False)
    main(True, True, True)

import sys
from pathlib import Path
import math

import numpy as np
import trimesh

import torch
from torchvision import transforms as tv_transforms

from diffaqua import transforms as fish_transforms
from diffpd.mesh import MeshHex

import pyvista as pv


def get_muscle_density(target_size, muscle_matrix):
    if (muscle_matrix[2] == 0.0).all():
        xyz = np.stack(np.meshgrid(*[np.arange(t) for t in target_size], indexing='ij'), axis=-1)
        xyz = np.array(xyz, dtype=np.float64)
        xyz += 0.5
        xyz -= muscle_matrix[1]
        xyz /= muscle_matrix[0]
        xyz = np.square(xyz)
        xyz = -xyz
        xyz = np.exp(xyz)
        density = np.mean(xyz, axis=-1)
        xyz_mask = np.array(np.sum(xyz > 0.5, axis=-1) == 3, dtype=np.float64)
        density = xyz_mask * density
    else:
        roll, pitch, yaw = muscle_matrix[2]
        c_r, s_r = np.cos(roll), np.sin(roll)
        c_p, s_p = np.cos(pitch), np.sin(pitch)
        c_y, s_y = np.cos(yaw), np.sin(yaw)

        R_roll = np.eye(3)
        R_pitch = np.eye(3)
        R_yaw = np.eye(3)

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

        R = R_yaw @ (R_pitch @ R_roll)

        sigma = np.diag(muscle_matrix[0])
        sigma = R @ sigma @ R.T

        xyz = np.stack(np.meshgrid(*[np.arange(t) for t in target_size], indexing='ij'), axis=-1)
        xyz = np.array(xyz, dtype=np.float64)
        xyz += 0.5
        xyz -= muscle_matrix[1]

        xyz_shape = xyz.shape
        xyz_sigma = (xyz.reshape((-1, 3)) @ np.linalg.inv(sigma)).reshape(xyz_shape)
        xyz = np.sum(xyz_sigma * xyz, axis=-1)

        xyz = -xyz
        density = np.exp(xyz)

        density[density < 0.5] = 0.0

    return density


def get_model(name, **kwargs):
    ckpt = get_model_(name, **kwargs)
    print('Model {} loaded with size {}'.format(name, ckpt['basis'].size()))
    return ckpt


def get_model_(name, **kwargs):
    root_path = Path(__file__).resolve().parent
    mesh_path = root_path / 'asset' / '3d'
    processed_root = mesh_path / 'processed'
    processed_root.mkdir(parents=True, exist_ok=True)
    load_preprocess = kwargs.pop('load_preprocess', True)

    sigma_base = 1 / math.sqrt(math.log(2))

    if name == 'lamprey':

        width = kwargs.pop('width')
        height = kwargs.pop('height')
        length = kwargs.pop('length')
        processed_path = processed_root / f'lamprey_{width}_{height}_{length}.pth'

        if load_preprocess and processed_path.is_file():
            return torch.load(processed_path)

        if width % 2 != 0 or height % 2 != 0:
            raise RuntimeError('Please make width and height even numbers!')

        voxel_transform = tv_transforms.Compose([
            fish_transforms.CreateChannel(),
            fish_transforms.ProbNormalize('mean')])

        max_hw = max(height, width)
        left = int(abs(height - width) / 2)
        right = int(abs(height + width) / 2)
        voxel = np.zeros((length, max_hw, max_hw))
        if height >= width:
            voxel[:, left: right, :] = 1.0
        else:
            voxel[:, :, left: right] = 1.0

        target_size = voxel.shape

        muscle = torch.Tensor([[
            [sigma_base * length / 2, sigma_base * (max_hw / 2), sigma_base * (max_hw / 2)], 
            [length / 2, max_hw / 2, max_hw / 2], 
            [0.0, 0.0, 0.0], 
        ]])

    basis = torch.Tensor(voxel)
    basis = voxel_transform(basis)
    ckpt = dict(voxel=voxel, basis=basis, muscle=muscle)

    if load_preprocess:
        torch.save(ckpt, processed_path)

    else:
        np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)
        body_np = basis.squeeze().sum(0).int().numpy().transpose((1, 0))
        body_np = np.vstack([np.arange(body_np.shape[1]).reshape((1, -1)), body_np])
        body_np = np.hstack([np.arange(-1, body_np.shape[0] - 1).reshape((-1, 1)), body_np])
        print(body_np)

        print(voxel.shape)
        rest_mesh = MeshHex.load(basis.squeeze().div(basis.max()).round().numpy(), dx=1.0)
        mesh = rest_mesh.get_pv_boundary()

        muscle_meshes = []
        for m in muscle:
            muscle_voxel = np.around(get_muscle_density(target_size, m.numpy()))
            muscle_voxel *= rest_mesh.voxel
            rest_muscle_mesh = MeshHex.load(muscle_voxel, dx=1.0)
            muscle_mesh = rest_muscle_mesh.get_pv_boundary()
            muscle_meshes.append(muscle_mesh)

        path = None
        try:
            pv.set_plot_theme('document')
            plotter = pv.Plotter(off_screen=path is not None)
            plotter.show_grid()
            plotter.add_mesh(mesh, color='w', show_edges=True, opacity=0.2)
            for m in muscle_meshes:
                plotter.add_mesh(m, color='r', show_edges=True, opacity=0.2)
            plotter.show(screenshot=str(path) if path is not None else False)
        finally:
            plotter.clear()
            plotter.deep_clean()
            plotter.close()

    return ckpt

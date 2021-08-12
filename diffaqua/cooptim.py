from pathlib import Path
import math
from itertools import product
from collections import deque
import time

from tqdm import tqdm, trange

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

from diffpd.fem import DeformableHex, HydrodynamicsStateForceHex
from diffpd.sim import Sim
from diffpd import transforms as pd_transforms
from diffpd.mesh import MeshHex

import pyvista as pv
import imageio

from .nn import VoxelToElementSchlick


class Cooptimizer(object):
    def __init__(
            self,
            shape,
            controller,
            target_dir,
            method,
            options,
            backward_options,
            dtype,
            num_epochs,
            names,
            shape_lr=1e-3,
            controller_lr=1e-3,
            controller_wd=1e-4,
            momentum=0.5,
            dx=0.02,
            muscle_stiffness=5e5,
            average_density=1e3,
            w_head=0.0,
            w_forward=1.0,
            w_sideward=10.0,
            w_efficiency=0.1,
            dt=3.33e-2,
            num_frames=200,
            tensorboard=None,
            plotter=True,
            pos_len=10,
            eps=1e-7,
            v_water=(0, 0, 0),
            youngs_modulus=5e6,
            poissons_ratio=4.9,
            schlick_a=0.1,
            schlick_alpha=0.5,
    ):

        self.base_youngs_modulus = youngs_modulus
        self.base_poissons_ratio = poissons_ratio
        self.v2e = VoxelToElementSchlick(schlick_a, alpha=schlick_alpha, cooptimizer=self)

        self.eps = eps
        self.names = names

        self.shape = shape
        self.controller = controller
        self.target_dir = target_dir.detach()

        self.dx = dx
        self.muscle_stiffness = muscle_stiffness
        self.average_density = average_density
        self.method = method
        self.options = options
        self.backward_options = backward_options
        self.w_head = w_head
        self.w_forward = w_forward
        self.w_sideward = w_sideward
        self.w_efficiency = w_efficiency
        self.dt = dt
        self.num_frames = num_frames

        self.dtype = dtype

        self.pos_len = pos_len

        self.shape_optimizer = optim.Adam(
            self.shape.parameters(), lr=shape_lr)

        self.controller_optimizer = optim.Adam(
            self.controller.parameters(),
            lr=controller_lr, weight_decay=controller_wd)

        self.joint_optimizer = optim.Adam(
            [
                {
                    'params': list(self.controller.parameters()),
                    'lr': controller_lr,
                    'weight_decay': controller_wd
                }, {
                    'params': list(self.shape.parameters()),
                    'lr': shape_lr,
                    'weight_decay': 0.0
                },
            ],
            lr=controller_lr)

        self.shape_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.shape_optimizer, lambda epoch: 1 - epoch / num_epochs)
        self.controller_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.controller_optimizer, lambda epoch: 1 - epoch / num_epochs)
        self.joint_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.joint_optimizer, lambda epoch: 1 - epoch / num_epochs)

        self.center = None
        self.mid_line = None
        self.mid_line_start = None
        self.mid_line_end = None

        self.rest_mesh = None
        self.deformable = None
        self.sim = None
        self.q0 = None
        self.v0 = None

        self.rest_mesh_blob = None
        self.rest_mesh_fish = None
        self.fish_to_blob_cell = None
        self.blob_to_fish_cell = None
        self.fish_to_blob_node = None
        self.blob_to_fish_node = None
        self.rest_mesh_fish_elements = None

        self.hydro = None
        self.tensorboard = tensorboard
        self.v_water = v_water

        self.controller_step = 0
        self.shape_step = 0

        self.plotter = pv.Plotter(off_screen=True) if plotter else None

        self.Cd_points = np.array([
            [0.0, 0.05],
            [0.4, 0.05],
            [0.7, 1.85],
            [1.0, 2.05],
        ])
        self.Ct_points = np.array([
            [-1, -0.8],
            [-0.3, -0.5],
            [0.3, 0.1],
            [1, 2.5],
        ])

        self.barycenter_elements = None
        self.cell_size = None
        self.node_size = None
        self.spine = None

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
            blob_boundary.T,
            head=spine[0],
            tail=spine[1]))
        self.hydro = HydrodynamicsStateForceHex(
            rho, v_water, self.Cd_points, self.Ct_points, fish_boundary.T)
        self.hydro.set_spine(
            head=blob_to_fish_node[spine[0]],
            tail=blob_to_fish_node[spine[1]])

        muscle_stiffness = self.muscle_stiffness
        all_muscles = []

        tail_muscle = muscles[0]
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

        if len(muscles) > 1:
            left_wing = muscles[1]

            wing_layers= [deque([None, None, None], maxlen=4) for _ in range(4)]

            z_cross = left_wing.sum(2)
            for z, wing_layer in zip(range(z_mid - 1, z_mid + 3), wing_layers):
                for xy in range(X + Y - 2):
                    y_start = min(xy, Y - 1)
                    x_start = xy - y_start

                    left_indices = []
                    right_indices = []
                    for step in range(min(y_start + 1, X - x_start)):
                        x = x_start + step
                        y = y_start - step
                        if z_cross[x, y] == 0:
                            continue
                        left_blob_idx = int(rest_mesh_blob.cell_indices[x, y, z])
                        fish_idx = int(rest_mesh_fish.cell_indices[x, y, z])
                        if fish_idx == -1 or left_wing[x, y, z] == 0:
                            continue
                        left_indices.append(left_blob_idx)
                        right_indices.append(int(rest_mesh_blob.cell_indices[x, Y - 1 - y, z]))

                    if len(left_indices) < 3:
                        continue

                    shared_muscles = [[left_indices], [right_indices]]
                    wing_layer.append(shared_muscles)

            wing_muscles = [shared_muscles for wing_layer in wing_layers for shared_muscles in wing_layer]
            for shared_muscles in wing_muscles:
                if shared_muscles is None:
                    continue
                transform.append(pd_transforms.AddActuationEnergy(
                    muscle_stiffness * 2, [-2 / math.sqrt(2), 2 / math.sqrt(2), 0.0], shared_muscles[0][0]))
                transform.append(pd_transforms.AddActuationEnergy(
                    muscle_stiffness * 2, [2 / math.sqrt(2), 2 / math.sqrt(2), 0.0], shared_muscles[1][0]))
            all_muscles.extend(wing_muscles)

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

        torch.save({
            'all_muscle': all_muscles,
            'fish_to_blob_cell': fish_to_blob_cell,
            'blob_to_fish_cell': blob_to_fish_cell,
            'fish_to_blob_node': fish_to_blob_node,
            'blob_to_fish_node': blob_to_fish_node,
        }, path / 'muscle.pth')

    def add_pd_energies(self, energy_types, youngs_modulus, poissons_ratio):
        return self.sim.add_default_pd_energies(energy_types, youngs_modulus, poissons_ratio)

    def get_state(self, q, v):
        q_center = q.view(-1, 3)[self.center]
        v_center = v.view(-1, 3)[self.center]

        q_mid_line_start_rel = q.view(-1, 3)[self.mid_line_start] - q_center.detach()
        q_mid_line_end_rel = q.view(-1, 3)[self.mid_line_end] - q_center.detach()
        v_mid_line_start = v.view(-1, 3)[self.mid_line_start]
        v_mid_line_end = v.view(-1, 3)[self.mid_line_end]
        state = [
            v_center,
            q_mid_line_start_rel.view(-1),
            q_mid_line_end_rel.view(-1),
            v_mid_line_start.view(-1),
            v_mid_line_end.view(-1),
        ]
        return torch.cat(state).unsqueeze(0)

    def get_loss_closed(self, **kwargs):
        path = kwargs.pop('path', None)

        if path is not None:
            path.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(path / 'video.mp4', fps=20)

        a = None
        q, v = self.q0, self.v0
        get_state = self.get_state
        controller = self.controller
        sim = self.sim
        mid_line = self.mid_line
        target_dir = self.target_dir

        forward_loss = torch.zeros(1)
        sideward_loss = torch.zeros(1)
        head_loss = torch.zeros(1)

        period = 25

        pos_base = torch.pow(2, torch.arange(self.pos_len, dtype=self.dtype)).unsqueeze(0).detach()

        for frame in trange(self.num_frames):
            state = get_state(q, v)

            pos_idx = math.pi * (frame % period) / period
            pos_sin = torch.sin(pos_base * pos_idx)
            pos_cos = torch.cos(pos_base * pos_idx)

            state = torch.cat([state, pos_sin, pos_cos], dim=1)

            a = controller(state, a)
            q, v = sim(q, v, a, **kwargs)

            v_zero = torch.zeros_like(v).view(-1, 3)
            v_zero[:, :-1] = v.view(-1, 3)[:, :-1]
            v_zero = v_zero.view(-1)

            v_center = v_zero.view(-1, 3)[mid_line].mean(dim=0)

            dot = torch.dot(v_center, target_dir)
            forward_loss += -dot

            cross = torch.cross(v_center, target_dir)
            sideward_loss += torch.dot(cross, cross)

            head_dir = q.view(-1, 3)[self.spine[0]] - q.view(-1, 3)[self.spine[1]]
            head_dir = head_dir / head_dir.norm()
            head_dot = torch.dot(head_dir, target_dir)
            head_loss += -head_dot

            if path is not None:
                img, rest_mesh = self.plot(q, v, v_center=v_center)
                q_save = q.view(self.node_size + (3,)).detach().numpy()
                v_save = v.view(self.node_size + (3,)).detach().numpy()
                a_save = a.detach().numpy()
                writer.append_data(img)
                np.savez_compressed(path / f'{frame:04d}.npz',
                    q=q_save, v=v_save, a=a_save, **rest_mesh.state_dict())

            v = v_zero

        if path is not None:
            writer.close()

        print(forward_loss.item() * self.w_forward, sideward_loss.item() * self.w_sideward, head_loss.item() * self.w_head)

        return self.w_forward * forward_loss + self.w_sideward * sideward_loss + self.w_head * head_loss

    def get_loss_open(self, **kwargs):
        path = kwargs.pop('path', None)

        if path is not None:
            path.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(path / 'video.mp4', fps=20)

        a = None
        q, v = self.q0, self.v0
        controller = self.controller
        sim = self.sim
        mid_line = self.mid_line
        target_dir = self.target_dir

        forward_loss = 0
        sideward_loss = 0

        for frame, a in tqdm(enumerate(controller()), total=self.num_frames):
            q, v = sim(q, v, a, **kwargs)

            v_zero = torch.zeros_like(v).view(-1, 3)
            v_zero[:, :-1] = v.view(-1, 3)[:, :-1]
            v_zero = v_zero.view(-1)

            v_center = v_zero.view(-1, 3)[mid_line].mean(dim=0)

            dot = torch.dot(v_center, target_dir)
            forward_loss += -dot

            cross = torch.cross(v_center, target_dir)
            sideward_loss += torch.dot(cross, cross)

            if path is not None:
                img, rest_mesh = self.plot(q, v, v_center=v_center)
                q_save = q.view(self.node_size + (3,)).detach().numpy()
                v_save = v.view(self.node_size + (3,)).detach().numpy()
                a_save = a.detach().numpy()
                writer.append_data(img)
                np.savez_compressed(path / f'{frame:04d}.npz',
                    q=q_save, v=v_save, a=a_save, **rest_mesh.state_dict())

            v = v_zero

        if path is not None:
            writer.close()

        return forward_loss + self.w_sideward * sideward_loss

    def train_joint(self, plot_path=None, ckpt_path=None, openloop_flag=False):
        self.shape_step += 1

        barycenter, _, muscles = self.shape()

        muscles_densities = []
        for muscle in muscles:
            muscles_density = self.shape.muscle_density(barycenter, muscle)
            muscles_densities.append(muscles_density)

        self.precompute(barycenter, muscles_densities, plot_path / f'{self.shape_step:04d}')

        barycenter_elements = self.v2e(barycenter)
        self.barycenter_elements = barycenter_elements.clone().detach().cpu().numpy()
        youngs_modulus = barycenter_elements * self.base_youngs_modulus + 1
        poissons_ratio = torch.full((self.rest_mesh_blob.num_elements,), self.base_poissons_ratio).detach()

        w = self.add_pd_energies(['corotated'], youngs_modulus, poissons_ratio)

        if not openloop_flag:
            loss = self.get_loss_closed(w=w, path=plot_path / f'{self.shape_step:04d}')
        else:
            loss = self.get_loss_open(w=w, path=plot_path / f'{self.shape_step:04d}')
        print('Joint:', loss.item())
        self.joint_optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            alpha_grad = self.shape.alpha.grad.data
            alpha_grad -= alpha_grad.mean()
            self.shape.alpha.grad.data.copy_(alpha_grad)
        nn.utils.clip_grad_norm_(self.shape.parameters(), 1.0)
        self.joint_optimizer.step()
        with torch.no_grad():
            alpha = self.shape.alpha.data
            if alpha.min() < 0:
                alpha.sub_(alpha.min())
            alpha.div_(alpha.sum() + self.eps)
            self.shape.alpha.data.copy_(alpha)

        if openloop_flag:
            self.controller.magnitudes.data.clamp_(0.0, 1.0)

        ckpt = {
            'shape': self.shape.state_dict(),
            'controller': self.controller.state_dict()
        }
        torch.save(ckpt, ckpt_path / f'{self.shape_step:04d}.pth')

        if self.tensorboard is not None:
            self.tensorboard.add_scalar(
                'overview/controller',
                loss.item(),
                self.shape_step)
            self.tensorboard.add_scalar(
                'overview/shape',
                loss.item(),
                self.shape_step)
            self.tensorboard.add_scalar(
                'overview/total',
                loss.item(),
                self.shape_step)

            full_alpha = self.shape.alpha.data.clone().detach().cpu().numpy()
            for i, a in enumerate(full_alpha):
                self.tensorboard.add_scalar(f'shape/{i}', a, self.shape_step)

    def train_controller(self, plot_path=None, ckpt_path=None, openloop_flag=False):
        self.controller_step += 1

        with torch.no_grad():
            barycenter, _, muscles = self.shape()

            muscles_densities = []
            for muscle in muscles:
                muscles_density = self.shape.muscle_density(barycenter, muscle)
                muscles_densities.append(muscles_density)

            self.precompute(barycenter, muscles_densities, plot_path / f'{self.controller_step + self.shape_step:04d}')

            barycenter_elements = self.v2e(barycenter)
            self.barycenter_elements = barycenter_elements.clone().detach().cpu().numpy()
            youngs_modulus = barycenter_elements * self.base_youngs_modulus + 1
            poissons_ratio = torch.full((self.rest_mesh_blob.num_elements,), self.base_poissons_ratio)
            self.add_pd_energies(['corotated'], youngs_modulus, poissons_ratio)

        self.controller.train()

        if not openloop_flag:
            loss = self.get_loss_closed(path=plot_path / f'{self.controller_step + self.shape_step:04d}')
        else:
            loss = self.get_loss_open(path=plot_path / f'{self.controller_step + self.shape_step:04d}')
        print('Controller:', loss.item())
        self.controller_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.controller.parameters(), 1.0)
        self.controller_optimizer.step()

        if openloop_flag:
            self.controller.magnitudes.data.clamp_(0.0, 1.0)

        ckpt = {
            'shape': self.shape.state_dict(),
            'controller': self.controller.state_dict()
        }
        torch.save(ckpt, ckpt_path / f'{self.controller_step + self.shape_step:04d}.pth')

        if self.tensorboard is not None:
            self.tensorboard.add_scalar(
                'overview/controller',
                loss.item(),
                self.controller_step)
            self.tensorboard.add_scalar(
                'overview/total',
                loss.item(),
                self.controller_step + self.shape_step)

    def train_shape(self, plot_path=None, ckpt_path=None, openloop_flag=False):
        self.shape_step += 1

        barycenter, _, muscles = self.shape()

        muscles_densities = []
        for muscle in muscles:
            muscles_density = self.shape.muscle_density(barycenter, muscle)
            muscles_densities.append(muscles_density)

        self.precompute(barycenter, muscles_densities, plot_path / f'{self.controller_step + self.shape_step:04d}')

        barycenter_elements = self.v2e(barycenter)
        self.barycenter_elements = barycenter_elements.clone().detach().cpu().numpy()
        youngs_modulus = barycenter_elements * self.base_youngs_modulus + 1
        poissons_ratio = torch.full((self.rest_mesh_blob.num_elements,), self.base_poissons_ratio).detach()

        w = self.add_pd_energies(['corotated'], youngs_modulus, poissons_ratio)

        if not openloop_flag:
            loss = self.get_loss_closed(w=w, path=plot_path / f'{self.controller_step + self.shape_step:04d}')
        else:
            loss = self.get_loss_open(w=w, path=plot_path / f'{self.controller_step + self.shape_step:04d}')

        self.shape_optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            alpha_grad = self.shape.alpha.grad.data.clone()
            alpha_grad -= alpha_grad.mean()
            self.shape.alpha.grad.data.copy_(alpha_grad)
            print(alpha_grad)
        nn.utils.clip_grad_norm_(self.shape.parameters(), 1.0)
        self.shape_optimizer.step()
        with torch.no_grad():
            alpha = self.shape.alpha.data
            if alpha.min() < 0:
                alpha.sub_(alpha.min())
            alpha.div_(alpha.sum() + self.eps)
            self.shape.alpha.data.copy_(alpha)

        ckpt = {
            'shape': self.shape.state_dict(),
            'controller': self.controller.state_dict()
        }
        torch.save(ckpt, ckpt_path / f'{self.controller_step + self.shape_step:04d}.pth')

        print('Shape:', loss.item(),
            self.shape.alpha.data.detach().cpu().numpy())

        if self.tensorboard is not None:
            self.tensorboard.add_scalar(
                'overview/shape',
                loss.item(),
                self.shape_step)
            self.tensorboard.add_scalar(
                'overview/total',
                loss.item(),
                self.controller_step + self.shape_step)

            full_alpha = self.shape.alpha.data.clone().detach().cpu().numpy()
            for i, a in enumerate(full_alpha):
                self.tensorboard.add_scalar(f'shape/{i}', a, self.shape_step)

    @torch.no_grad()
    def plot_top(self, q, v, **kwargs):
        blob_q = q.detach().cpu().numpy().reshape((-1, 3))
        fish_q = np.zeros_like(self.rest_mesh_fish.vertices)
        for k, val in sorted(self.fish_to_blob_node.items()):
            fish_q[k] = blob_q[val]
        rest_mesh = MeshHex(fish_q, self.rest_mesh_fish.elements)
        mesh = rest_mesh.get_pv_boundary()
        if self.barycenter_elements is not None:
            _, indices = rest_mesh.get_color_boundary()
            scalars = np.array([self.barycenter_elements[self.fish_to_blob_cell[i]] for i in indices])
        else:
            scalars = None

        try:
            self.plotter.add_mesh(mesh, scalars=scalars, color='w', clim=[0.5, 1], show_edges=True, name='fish')
            self.plotter.add_scalar_bar()
            if 'v_center' in kwargs:
                mean_v = kwargs['v_center'].numpy()
                msg = 'Vx:{:.3f}\nVy:{:.3f}\nVz:{:.3f}'.format(*mean_v)
                self.plotter.add_text(msg, position='upper_edge', name='mean_v', font_size=10)
            msg = '\n'.join(['{}: {:.2f}%'.format(' '.join(name.split('_')).title(), alpha * 100) for name, alpha in zip(
                self.names, self.shape.alpha.data.clone().detach().cpu().numpy())])
            self.plotter.add_text(msg, position='left_edge', name='interp', font_size=10)
            self.plotter.camera_position = [
                (-0.5, 0.0, 5.0),
                (-0.5, 0.0, 0.0),
                (0.0, 1.0, 0.0)]
            _, img = self.plotter.show(screenshot=True, return_img=True, auto_close=False)
        finally:
            self.plotter.clear()
        return img, rest_mesh

    @torch.no_grad()
    def plot(self, q, v, **kwargs):
        blob_q = q.detach().cpu().numpy().reshape((-1, 3))
        fish_q = np.zeros_like(self.rest_mesh_fish.vertices)
        for k, val in sorted(self.fish_to_blob_node.items()):
            fish_q[k] = blob_q[val]
        rest_mesh = MeshHex(fish_q, self.rest_mesh_fish.elements)
        mesh = rest_mesh.get_pv_boundary()
        if self.barycenter_elements is not None:
            _, indices = rest_mesh.get_color_boundary()
            scalars = np.array([self.barycenter_elements[self.fish_to_blob_cell[i]] for i in indices])
        else:
            scalars = None

        try:
            self.plotter.add_mesh(mesh, scalars=scalars, color='w', clim=[0.5, 1], show_edges=True, name='fish')
            self.plotter.add_scalar_bar()
            if 'v_center' in kwargs:
                mean_v = kwargs['v_center'].numpy()
                msg = 'Vx:{:.3f}\nVy:{:.3f}\nVz:{:.3f}'.format(*mean_v)
                self.plotter.add_text(msg, position='upper_edge', name='mean_v', font_size=10)
            msg = '\n'.join(['{}: {:.2f}%'.format(' '.join(name.split('_')).title(), alpha * 100) for name, alpha in zip(
                self.names, self.shape.alpha.data.clone().detach().cpu().numpy())])
            self.plotter.add_text(msg, position='left_edge', name='interp', font_size=10)
            self.plotter.camera_position = [
                (-0.5, -3.0, 1.2),
                (-0.5, 0.0, 0.0),
                (0.0, 0.0, 1.0)]
            _, img = self.plotter.show(screenshot=True, return_img=True, auto_close=False)
        finally:
            self.plotter.clear()
        return img, rest_mesh

    @torch.no_grad()
    def plot_blob(self, q):
        rest_mesh = MeshHex(q.detach().cpu().numpy(), self.rest_mesh_blob.elements)
        mesh = rest_mesh.get_pv_boundary()

        try:
            self.plotter.add_mesh(mesh, color='w', show_edges=True, name='fish')
            
            self.plotter.camera_position = [
                (-0.5, -3.0, 1.2),
                (-0.5, 0.0, 0.0),
                (0.0, 0.0, 1.0)]
            _, img = self.plotter.show(screenshot=True, return_img=True, auto_close=False)
        finally:
            self.plotter.clear()
        return img, rest_mesh

    @torch.no_grad()
    def plot_arrow(self, q, v):
        blob_q = q.detach().cpu().numpy().reshape((-1, 3))
        fish_q = np.zeros_like(self.rest_mesh_fish.vertices)
        for k, val in sorted(self.fish_to_blob_node.items()):
            fish_q[k] = blob_q[val]
        blob_v = v.detach().cpu().numpy().reshape((-1, 3))
        fish_v = np.zeros_like(self.rest_mesh_fish.vertices)
        for k, val in sorted(self.fish_to_blob_node.items()):
            fish_v[k] = blob_v[val]
        rest_mesh = MeshHex(fish_q, self.rest_mesh_fish.elements)
        mesh = rest_mesh.get_pv_boundary()

        hydroforce = self.hydro.forward(fish_q.ravel(), fish_v.ravel()).reshape((-1, 3))
        surface = list(sorted(set(rest_mesh.boundary.ravel())))

        arrow_cent = np.vstack([fish_q[i] for i in surface])
        arrow_dir = np.vstack([hydroforce[i] for i in surface])

        try:
            self.plotter.add_mesh(mesh, color='w', show_edges=True, name='fish')
            self.plotter.add_arrows(arrow_cent, arrow_dir)
            self.plotter.camera_position = [
                (-0.5, -3.0, 1.2),
                (-0.5, 0.0, 0.0),
                (0.0, 0.0, 1.0)]
            _, img = self.plotter.show(screenshot=True, return_img=True, auto_close=False)
        finally:
            self.plotter.clear()
        return img, rest_mesh

    def __del__(self):
        self.plotter.deep_clean()
        self.plotter.close()

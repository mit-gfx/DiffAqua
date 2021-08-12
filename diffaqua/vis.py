from pathlib import Path
from itertools import product
import math

import numpy as np
import imageio

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
import matplotlib.animation as animation

import plotly.graph_objects as go

import torch


class Visualizer3d(object):
    def __init__(self, save_root, grid_size):
        self.save_root = save_root
        self.x, self.y, self.z = np.mgrid[:grid_size, :grid_size, :grid_size]
        self.fig = go.Figure(layout=dict(margin=dict(t=0, l=0, b=0))) 

    def plot(self, voxel, vector_size=0):
        voxel = voxel.contiguous().clone().detach().cpu().numpy()
        self.fig.update(data=go.Isosurface(
            x=self.x.flatten(),
            y=self.y.flatten(),
            z=self.z.flatten(),
            value=voxel.flatten(),
            isomin=1 - 0.1,
            isomax=(vector_size + 1) + 0.1,
            surface_count=vector_size + 2,
            opacity=0.6,
            caps=dict(x_show=False, y_show=False)
        ), overwrite=True)

    def save(self, filename, write_image=False):
        self.fig.write_html(str(self.save_root / (filename + '.html')), auto_open=False)

        if write_image:
            self.fig.write_image(str(self.save_root / (filename + '.jpg')), width=800, height=600, engine='orca')
        self.fig.update(data=None, overwrite=True)


class Visualizer2d(object):
    def __init__(self, save_root, figsize=(224, 224)):
        self.save_root = save_root
        self.figsize = figsize

        self.plt_kit = None

        self.fig, self.ax = plt.subplots(1, 1, figsize=figsize, dpi=1)
        self.ax.axis('off')

    def plot(self, image, cmap='gray'):
        if self.plt_kit is None:
            self.plt_kit = plt.subplots(1, 1, figsize=self.figsize, dpi=1)
            self.plt_kit[1].axis('off')

        image = image.contiguous().clone().detach().cpu().numpy()
        self.plt_kit[1].imshow(image, cmap)

    def save(self, filename, show=False):
        self.plt_kit[0].tight_layout()

        if show:
            plt.show()

        self.plt_kit[0].savefig(self.save_root / (filename + '.jpg'))
        plt.close(self.plt_kit[0])
        self.plt_kit = None


class Visualizer1d(object):
    def __init__(self, save_root, x, figsize=(10, 10)):
        self.save_root = save_root
        self.x = x
        self.figsize = figsize
        width = x[1:] - x[:-1]
        self.width = np.append(width, np.mean(width))

        self.plt_kit = None

    def plot(self, signal, color='tab:blue', alpha=1.0, zorder=1.0):
        if self.plt_kit is None:
            self.plt_kit = plt.subplots(1, 1, figsize=self.figsize)

        signal = signal.contiguous().clone().detach().cpu().numpy()
        self.plt_kit[1].bar(self.x, signal, width=self.width, alpha=alpha, color=color, zorder=zorder)

    def save(self, filename, show=False):
        self.plt_kit[0].tight_layout()

        if show:
            plt.show()

        self.plt_kit[0].savefig(self.save_root / (filename + '.jpg'))
        plt.close(self.plt_kit[0])
        self.plt_kit = None


class Arrow3D(FancyArrowPatch):
    def __init__(self, xyz, dxdydz, *args, **kwargs):
        super(Arrow3D, self).__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = xyz
        self._dxdydz = dxdydz

    def set_positions(self, xyz, dxdydz): 
        self._xyz = xyz
        self._dxdydz = dxdydz

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), renderer.M)
        super(Arrow3D, self).set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super(Arrow3D, self).draw(renderer)


def _arrow3D(ax, xyz, dxdydz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''
    arrow = Arrow3D(xyz, dxdydz, *args, **kwargs)
    ax.add_artist(arrow)
    return arrow
setattr(Axes3D, 'arrow3D', _arrow3D)


class Fish3DVisualizer(object):
    def __init__(self, sim, controller, get_state, get_plot, init_qv, num_frames, dx):
        self.sim = sim
        self.controller = controller
        self.get_state = get_state
        self.get_plot = get_plot
        self.init_qv = init_qv
        self.num_frames = num_frames
        self.dx = dx

        self.stream = None
        self.fig, self.ax = None, None

        self.scatter = None
        self.arrow_target = None
        self.arrow_vel = None
        self.title = None

    def save(self, filename):
        fps = 10
        self.fig, self.ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw=dict(projection='3d'))
        ani = animation.FuncAnimation(
            self.fig, self.update, self.num_frames, init_func=self.setup_plot, interval=1 / fps, repeat=False, blit=False)
        writer = animation.writers['ffmpeg'](fps=fps)
        ani.save(filename, writer=writer)
        plt.close(self.fig)


    def setup_plot(self):
        self.stream = self.data_stream()
        q, v_center, arrow_target_data = next(self.stream)
        xs, ys, zs = q.numpy().T

        self.scatter = self.ax.scatter(xs, ys, zs, c='tab:blue')

        self.arrow_target = self.ax.arrow3D(
            (0, 0, 1 * self.dx),
            arrow_target_data.numpy() * self.dx,
            mutation_scale=10,
            ec='tab:red', fc='tab:red')

        self.arrow_vel = self.ax.arrow3D(
            (0, 0, 1 * self.dx),
            v_center.numpy(),
            mutation_scale=10,
            ec='tab:green', fc='tab:green')

        radius = 50
        self.ax.set_xlim([-radius * 1.5 * self.dx, radius * 0.5 * self.dx])
        self.ax.set_ylim([-radius * self.dx, radius * self.dx])
        self.ax.set_zlim([-radius * self.dx, radius * self.dx])
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.title = self.ax.set_title('Animation step=0?', animated=True)

        return self.scatter, self.arrow_target, self.arrow_vel, self.title

    def update(self, i):
        q, v_center, arrow_target_data = next(self.stream)
        xs, ys, zs = q.numpy().T

        self.scatter._offsets3d = (xs, ys, zs)  

        self.arrow_target.set_positions((0, 0, 1 * self.dx), arrow_target_data.numpy() * self.dx)
        self.arrow_vel.set_positions((0, 0, 1 * self.dx), v_center.numpy())

        self.title.set_text(f'Animation step={i}')

        return self.scatter, self.arrow_target, self.arrow_vel, self.title

    @torch.no_grad()
    def data_stream(self):

        self.controller.train(False)

        q, v = self.init_qv
        get_plot = self.get_plot
        get_state = self.get_state
        controller = self.controller
        sim = self.sim

        a = None

        for frame in range(1, self.num_frames + 1):

            yield get_plot(q, v)

            state = get_state(q, v)
            a = controller(state, a)
            q, v = sim(q=q, v=v, a=a)

        yield get_plot(q, v)

from typing import Iterable

import numpy as np
from gym import spaces

import grid_world.rendering.boolean_draw_functions as drawing
import utils.model_utils as model_utils


class ObservationRenderer():
    def __init__(self, cfg):
        self.resolution = cfg['global_resolution']

        self.obs_resolution = cfg.get('observation_resolution', self.resolution)
        assert model_utils.is_odd(self.obs_resolution[0]) and model_utils.is_odd(
            self.obs_resolution[1]), 'keep resolutions odd for simplicity'

        self.n_channels = cfg.get('n_channels', 1)
        self.drawing = np.zeros((self.n_channels, self.resolution[0], self.resolution[1]), dtype=np.uint8)

    def get_drawing(self):
        return self.drawing

    def reset_drawing(self):
        '''
        start a new drawing
        :return:
        '''
        self.drawing = np.zeros_like(self.drawing)

    def get_obs_shape(self):
        high = 1.
        low = 0.
        return spaces.Box(high=high, low=low, shape=(self.obs_resolution[0], self.obs_resolution[1]))

    def draw_circle(self, center: Iterable[int], radius: float = 5., channel: int = 0):
        self.drawing[channel] = drawing.draw_circle(self.drawing[channel], center=center, radius=radius)
        return self.drawing

    def draw_ring(self, center: Iterable[int], radius: float = 5., width: float = 5., channel: int = 0):
        self.drawing[channel] = drawing.draw_ring(self.drawing[channel], center=center, radius=radius, width=width)
        return self.drawing

    def draw_diamond(self, center: Iterable[int], apothem: float = 5., channel: int = 0):
        self.drawing[channel] = drawing.draw_diamond(self.drawing[channel], center=center, apothem=apothem)
        return self.drawing

    def get_frame_at_point(self, center: Iterable[int]):
        # todo https://shapely.readthedocs.io/en/latest/manual.html#object.intersection
        frame = np.zeros((self.n_channels, self.obs_resolution[0], self.obs_resolution[1]))
        frame_center = (self.obs_resolution[0] // 2, self.obs_resolution[1] // 2)
        row_start = int(max(0, center[0] - (frame_center[0] + 1)))
        col_start = int(max(0, center[1] - (frame_center[1] + 1)))
        row_end = int(min(self.obs_resolution[0], frame_center[0] + center[0] + 1))
        col_end = int(min(self.obs_resolution[1], frame_center[1] + center[1] + 1))

        drawing_slice = self.drawing[:, row_start:row_end, col_start:col_end]

        target_row_start = int(max(0, frame_center[0] - center[0]))
        target_col_start = int(max(0, frame_center[1] - center[1]))
        target_row_end = target_row_start + drawing_slice.shape[-2]
        target_col_end = target_col_start + drawing_slice.shape[-1]

        frame[:, target_row_start: target_row_end, target_col_start: target_col_end] = drawing_slice
        return frame

    def draw_line(self):
        raise NotImplementedError

    def convert_to_gif(self):
        pass


if __name__ == '__main__':
    renderer = EnvRenderer(resolution=100)
    circle = renderer.draw_circle(point=(1, 2), radius=100)
    print(circle.to())

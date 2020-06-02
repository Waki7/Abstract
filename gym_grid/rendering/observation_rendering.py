import numpy as np
from gym import spaces

import gym_grid.rendering.draw_functions as drawing


class ObservationRenderer():
    def __init__(self, cfg):
        self.resolution = cfg['global_resolution']
        self.obs_resolution = cfg.get('observation_resolution')
        self.n_channels = cfg.get('n_channels', 1)
        self.drawing = np.zeros((self.n_channels, self.resolution[0], self.resolution[1]), dtype=np.uint8)

    def get_drawing(self):
        return self.drawing

    def reset_drawing(self):
        '''
        start a new drawing
        :return:
        '''
        self.drawing = 0 * self.drawing

    def get_obs_shape(self):
        high = 1.
        low = 0.
        return spaces.Box(high=high, low=low, shape=(self.obs_resolution[0], self.obs_resolution[1]))

    def draw_circle(self, center: np.ndarray, radius: float = 5., channel: int = 0.):
        drawing.draw_circle(self.drawing[channel], center=center, radius=radius)

    def draw_ring(self, center: np.ndarray, radius: float = 5., width: float = 5., channel: int = 0):
        drawing.draw_ring(self.drawing[channel], center=center, radius=radius, width=width)

    def draw_square(self, center: np.ndarray, length: float = 5., channel: int = 0):
        drawing.draw_square(self.drawing[channel], center=center, length=length)

    def get_obs(self, center):
        pass

    def draw_line(self):
        raise NotImplementedError

    def convert_to_gif(self):
        pass


if __name__ == '__main__':
    renderer = EnvRenderer(resolution=100)
    circle = renderer.draw_circle(point=(1, 2), radius=100)
    print(circle.to())

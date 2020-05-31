import numpy as np

import gym_grid.rendering.draw_functions as drawing


class ObservationRenderer():
    def __init__(self, cfg):
        resolution = cfg['resolution']
        self.resolution = resolution
        self.drawing = np.zeros((resolution, resolution), dtype=np.uint8)

    def get_drawing(self):
        return self.drawing

    def start_drawing(self):
        '''
        start a new drawing
        :return:
        '''
        self.drawing = 0 * self.drawing

    def draw_circle(self, center: np.ndarray, radius):
        drawing.draw_circle(self.drawing, center=center, radius=radius)

    def draw_ring(self, center: np.ndarray, radius, width=10):
        drawing.draw_ring(self.drawing, center=center, radius=radius, width=width)

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

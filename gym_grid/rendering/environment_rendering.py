import numpy as np

from gym_grid.rendering.draw_functions import *


class EnvironmentRenderer():
    def __init__(self, resolution):
        self.resolution = resolution
        self.drawing = np.zeros((resolution, resolution), dtype=np.uint8)

    def get_drawing(self):
        return self.drawing




if __name__ == '__main__':
    renderer = EnvironmentRenderer(resolution=100)
    circle = renderer.draw_circle(point=(1, 2), radius=100)
    print(circle.to())

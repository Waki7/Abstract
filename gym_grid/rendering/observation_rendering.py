import numpy as np


class ObservationRenderer():
    def __init__(self, cfg):
        resolution = cfg['resolution']
        self.resolution = resolution
        self.drawing = np.zeros((resolution, resolution), dtype=np.uint8)

    def get_drawing(self):
        return self.drawing

    def draw_circle(self, center: np.ndarray, radius, width=10):
        '''
        Draws a circle on the Renderers current drawing memory, if width is not declared, will go with 1 pixel,
        see draw_circle method if you wanted a filled ring
        :param center: ndarray with two values, one for the y and one for the x dimension center
        :param radius: how many pixels from the center the ring should be drawn
        :param width: how many pixels wide the ring should be, default 1
        :return:
        '''
        xx, yy = np.mgrid[:self.resolution, :self.resolution]  # basically just a 2d np.arange
        circle_dist = (xx - center[1]) ** 2 + (yy - center[0]) ** 2
        half_width = width // 2
        ring_dist = radius ** 2
        circle = circle_dist < (ring_dist + half_width)
        return circle

    def draw_ring(self, center: np.ndarray, radius, width=10):
        '''
        Draws a circle on the Renderers current drawing memory, if width is not declared, will go with 1 pixel,
        see draw_circle method if you wanted a filled ring
        :param center: ndarray with two values, one for the y and one for the x dimension center
        :param radius: how many pixels from the center the ring should be drawn
        :param width: how many pixels wide the ring should be, default 1
        :return:
        '''
        xx, yy = np.mgrid[:self.resolution, :self.resolution]  # basically just a 2d np.arange
        circle_dist = (xx - center[1]) ** 2 + (yy - center[0]) ** 2
        half_width = width // 2
        ring_dist = radius ** 2
        ring = (circle_dist < (ring_dist + half_width)) & (circle_dist > (ring_dist - half_width))
        return ring

    def draw_line(self):
        raise NotImplementedError


if __name__ == '__main__':
    renderer = EnvRenderer(resolution=100)
    circle = renderer.draw_circle(point=(1, 2), radius=100)
    print(circle.to())

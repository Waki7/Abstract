from typing import Iterable

import numpy as np


class Shape():
    def __init__(self):
        pass

    def draw(self, background: np.ndarray, center: Iterable[float]):
        raise NotImplementedError


class Circle(Shape):
    def __init__(self, radius, observed_value=None):
        '''

        :param observed_value:  color to draw, expecting int 0-255, will only fill n channels in background
        :param radius: how many pixels from the center the ring should be drawn
        '''
        if not isinstance(observed_value, tuple):
            # always keep rgb, if we render in black and white then we can just take the first value
            observed_value = (observed_value, observed_value, observed_value,)
        self.value = observed_value
        self.radius = radius

    def boolean_circle(self, background: np.ndarray, center: Iterable[float]):
        yy, xx = np.mgrid[:background.shape[-2], :background.shape[-1]]  # basically just a 2d np.arange
        circle_dist = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
        ring_dist = self.radius ** 2
        circle = circle_dist < ring_dist
        return circle

    def draw(self, background: np.ndarray, center: Iterable[float]) -> np.ndarray:
        '''
        Draws a circle on the Renderers current drawing memory
        :param background: grid to draw over, will mimic shape
        :param center: ndarray with two values, one for the y and one for the x dimension center
        :return:
        '''
        circle = self.boolean_circle(background, center)
        if self.value is None:
            # draw in black/white using or function for overlap
            return np.logical_or(circle, background)
        for channel in range(0, background.shape[0]):
            background[channel, circle] = self.value[channel]
        return background


class Diamond(Shape):
    def __init__(self, apothem, observed_value=None):
        '''

        :param apothem: how many pixels from the of the diamond to a corner
        :param observed_value: color to draw, expecting int 0-255, will only fill n channels in background
        '''
        if not isinstance(observed_value, tuple):
            # always keep rgb, if we render in black and white then we can just take the first value
            observed_value = (observed_value, observed_value, observed_value,)
        self.value = observed_value
        self.apothem = apothem

    def boolean_diamond(self, background: np.ndarray, center: Iterable[float]):
        yy, xx = np.mgrid[:background.shape[-2], :background.shape[-1]]  # basically just a 2d np.arange
        center_dist = np.abs(yy - center[0]) + np.abs(xx - center[1])
        diamond = (center_dist <= self.apothem)
        return diamond

    def draw(self, background: np.ndarray, center: Iterable[float]) -> np.ndarray:
        # TODO THIS IS A DIAMOND BECAUSE OF THE CORNERS
        '''
            Draws a square on the Renderers current drawing memory
            :param background: grid to draw over, will mimic shape
            :param center: ndarray with two values, one for the y and one for the x dimension center
            :return:
            '''
        diamond = self.boolean_diamond(background, center)
        if self.value is None:
            # draw in black/white using or function for overlap
            return np.logical_or(diamond, background)
        for channel in range(0, background.shape[0]):
            background[channel, diamond] = self.value[channel]
        return background


class Ring(Shape):
    def __init__(self, radius, width, observed_value=None):
        '''

        :param radius: how many pixels from the center the ring should be drawn
        :param width: how many pixels wide the ring should be, default 1
        :param observed_value: color to draw, expecting int 0-255, will only fill n channels in background
        '''
        if not isinstance(observed_value, tuple):
            # always keep rgb, if we render in black and white then we can just take the first value
            observed_value = (observed_value, observed_value, observed_value,)
        self.value = observed_value
        self.radius = radius
        self.width = width

    def boolean_ring(self, background: np.ndarray, center: Iterable[float]):
        yy, xx = np.mgrid[:background.shape[-2], :background.shape[-1]]  # basically just a 2d np.arange
        circle_dist = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
        half_width = self.width // 2
        ring_dist = self.radius ** 2
        ring = (circle_dist < (ring_dist + half_width)) & (circle_dist > (ring_dist - half_width))
        return ring

    def draw_ring(self, background: np.ndarray, center: Iterable[float]):
        '''
        Draws a ring on the Renderers current drawing memory,
        see draw_circle method if you wanted a filled ring
        :param background: grid to draw over, will mimic shape
        :param center: ndarray with two values, one for the y and one for the x dimension center
        :return:
        '''
        ring = self.boolean_ring(background, center)
        if self.value is None:
            # draw in black/white using or function for overlap
            return np.logical_or(ring, background)
        for channel in range(0, background.shape[0]):
            background[channel, ring] = self.value[channel]
        return background

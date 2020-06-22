from typing import Iterable

import numpy as np


###
### DRAW FUNCTIONS DOESN'T DO ANYTHING SPECIAL FOR OVERLAPS
###

def draw_circle(background: np.ndarray, center: Iterable[float], radius):
    '''
    Draws a circle on the Renderers current drawing memory
    :param background: grid to draw over, will mimic shape
    :param center: ndarray with two values, one for the y and one for the x dimension center
    :param radius: how many pixels from the center the ring should be drawn
    :param width: how many pixels wide the ring should be, default 1
    :return:
    '''
    xx, yy = np.mgrid[:background.shape[0], :background.shape[1]]  # basically just a 2d np.arange
    circle_dist = (xx - center[1]) ** 2 + (yy - center[0]) ** 2
    ring_dist = radius ** 2
    circle = circle_dist < ring_dist
    return np.logical_or(circle, background)


def draw_ring(background: np.ndarray, center: Iterable[float], radius, width):
    '''
    Draws a ring on the Renderers current drawing memory,
    see draw_circle method if you wanted a filled ring
    :param background: grid to draw over, will mimic shape
    :param center: ndarray with two values, one for the y and one for the x dimension center
    :param radius: how many pixels from the center the ring should be drawn
    :param width: how many pixels wide the ring should be, default 1
    :return:
    '''
    xx, yy = np.mgrid[:background.shape[0], :background.shape[1]]  # basically just a 2d np.arange
    circle_dist = (xx - center[1]) ** 2 + (yy - center[0]) ** 2
    half_width = width // 2
    ring_dist = radius ** 2
    ring = (circle_dist < (ring_dist + half_width)) & (circle_dist > (ring_dist - half_width))
    return np.logical_or(ring, background)


def draw_diamond(background: np.ndarray, center: Iterable[float], apothem):
    # TODO THIS IS A DIAMOND BECAUSE OF THE CORNERS
    '''
        Draws a square on the Renderers current drawing memory
        :param background: grid to draw over, will mimic shape
        :param center: ndarray with two values, one for the y and one for the x dimension center
        :param apothem: how many pixels from the of the diamond to a corner
        :param width: how many pixels wide the ring should be, default 1
        :return:
        '''
    xx, yy = np.mgrid[:background.shape[0], :background.shape[1]]  # basically just a 2d np.arange
    center_dist = np.abs(xx - center[1]) + np.abs(yy - center[0])
    square = (center_dist <= apothem)
    return np.logical_or(square, background)


def draw_line(background):
    raise NotImplementedError

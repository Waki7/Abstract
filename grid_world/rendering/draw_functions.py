from typing import Iterable

import numpy as np


def draw_circle(background: np.ndarray, center: Iterable[int], radius):
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
    return circle + background


def draw_ring(background: np.ndarray, center: Iterable[int], radius, width):
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
    return ring + background


def draw_square(background: np.ndarray, center: Iterable[int], length):
    '''
        Draws a square on the Renderers current drawing memory
        :param background: grid to draw over, will mimic shape
        :param center: ndarray with two values, one for the y and one for the x dimension center
        :param radius: how many pixels from the center the ring should be drawn
        :param width: how many pixels wide the ring should be, default 1
        :return:
        '''
    xx, yy = np.mgrid[:background.shape[0], :background.shape[1]]  # basically just a 2d np.arange
    center_dist = (xx - center[1]) + (yy - center[0])
    half_length = length // 2
    ring = (center_dist < (center + half_length)) & (center_dist > (center - half_length))
    return ring + background


def draw_line(background):
    raise NotImplementedError

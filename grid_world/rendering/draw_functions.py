from typing import Iterable

import numpy as np


###
### DRAW FUNCTIONS DOESN'T DO ANYTHING SPECIAL FOR OVERLAPS
###

def boolean_circle(background: np.ndarray, center: Iterable[float], radius):
    yy, xx = np.mgrid[:background.shape[-2], :background.shape[-1]]  # basically just a 2d np.arange
    circle_dist = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
    ring_dist = radius ** 2
    circle = circle_dist < ring_dist
    return circle


def draw_circle(background: np.ndarray, center: Iterable[float], radius, value=None):
    '''
    Draws a circle on the Renderers current drawing memory
    :param value: color to draw, expecting int 0-255, will only fill n channels in background
    :param background: grid to draw over, will mimic shape
    :param center: ndarray with two values, one for the y and one for the x dimension center
    :param radius: how many pixels from the center the ring should be drawn
    :param width: how many pixels wide the ring should be, default 1
    :return:
    '''
    circle = boolean_circle(background, center, radius)
    if value is None:
        # draw in black/white using or function for overlap
        return np.logical_or(circle, background)
    for channel in range(0, background.shape[0]):
        background[channel, circle] = value[channel]
    return background


def boolean_ring(background: np.ndarray, center: Iterable[float], radius, width):
    yy, xx = np.mgrid[:background.shape[-2], :background.shape[-1]]  # basically just a 2d np.arange
    circle_dist = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
    half_width = width // 2
    ring_dist = radius ** 2
    ring = (circle_dist < (ring_dist + half_width)) & (circle_dist > (ring_dist - half_width))
    return ring


def draw_ring(background: np.ndarray, center: Iterable[float], radius, width, value=None):
    '''
    Draws a ring on the Renderers current drawing memory,
    see draw_circle method if you wanted a filled ring
    :param value: color to draw, expecting int 0-255, will only fill n channels in background
    :param background: grid to draw over, will mimic shape
    :param center: ndarray with two values, one for the y and one for the x dimension center
    :param radius: how many pixels from the center the ring should be drawn
    :param width: how many pixels wide the ring should be, default 1
    :return:
    '''
    ring = boolean_ring(background, center, radius, width)
    if value is None:
        # draw in black/white using or function for overlap
        return np.logical_or(ring, background)
    for channel in range(0, background.shape[0]):
        background[channel, ring] = value[channel]
    return background


def boolean_diamond(background: np.ndarray, center: Iterable[float], apothem):
    yy, xx = np.mgrid[:background.shape[-2], :background.shape[-1]]  # basically just a 2d np.arange
    center_dist = np.abs(yy - center[0]) + np.abs(xx - center[1])
    diamond = (center_dist <= apothem)
    return diamond


def draw_diamond(background: np.ndarray, center: Iterable[float], apothem, value=None):
    # TODO THIS IS A DIAMOND BECAUSE OF THE CORNERS
    '''
        Draws a square on the Renderers current drawing memory
        :param value: color to draw, expecting int 0-255, will only fill n channels in background
        :param background: grid to draw over, will mimic shape
        :param center: ndarray with two values, one for the y and one for the x dimension center
        :param apothem: how many pixels from the of the diamond to a corner
        :param width: how many pixels wide the ring should be, default 1
        :return:
        '''
    diamond = boolean_diamond(background, center, apothem)
    if value is None:
        # draw in black/white using or function for overlap
        return np.logical_or(diamond, background)
    for channel in range(0, background.shape[0]):
        background[channel, diamond] = value[channel]
    return background


def draw_line(background):
    raise NotImplementedError

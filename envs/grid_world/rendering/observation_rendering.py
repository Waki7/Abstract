from typing import Iterable

import numpy as np
from gym import spaces

from envs import grid_world as render_shapes
import utils.image_utils as image_utils
import utils.model_utils as model_utils


class ObservationRenderer():
    def __init__(self, cfg):
        self.global_resolution = cfg['global_resolution']

        # the default fallback obs_window permits full observation of the world
        obs_fallback_window = [int((2 * self.global_resolution[0]) - 1), int((2 * self.global_resolution[1]) - 1)]

        self.obs_window = cfg.get('observation_window', obs_fallback_window)
        self.obs_resolution = cfg.get('observation_resolution', self.global_resolution)
        assert model_utils.is_odd(self.obs_resolution[0]) and model_utils.is_odd(
            self.obs_resolution[1]), 'keep resolutions odd for simplicity'

        self.n_channels = cfg.get('n_channels', 3)
        self.observation_interpolation = cfg.get('observation_interpolation')

        self.drawing = np.zeros((self.n_channels, self.global_resolution[0], self.global_resolution[1]), dtype=np.uint8)

    def get_drawing(self):
        return self.drawing

    def reset_drawing(self):
        '''
        start a new drawing
        :return:
        '''
        self.drawing = np.zeros_like(self.drawing)

    def get_obs_shape(self):
        high = 255.
        low = 0.
        return spaces.Box(high=high, low=low, shape=(self.n_channels, self.obs_resolution[0], self.obs_resolution[1]))

    def draw_shape(self, shape: render_shapes.Shape, center: Iterable[float]):
        self.drawing = shape.draw(self.drawing, center=center)
        return self.drawing

    def get_frame_at_point(self, center: Iterable[float]):
        # todo https://shapely.readthedocs.io/en/latest/manual.html#object.intersection
        frame = np.zeros((self.n_channels, self.obs_window[0], self.obs_window[1]))
        frame_center = (self.obs_window[0] // 2, self.obs_window[1] // 2)
        row_start = int(max(0, center[0] - frame_center[0]))
        col_start = int(max(0, center[1] - frame_center[1]))
        row_end = int(min(self.global_resolution[0], frame_center[0] + center[0] + 1))
        col_end = int(min(self.global_resolution[1], frame_center[1] + center[1] + 1))

        drawing_slice = self.drawing[:, row_start:row_end, col_start:col_end]

        target_row_start = int(max(0, frame_center[0] - center[0]))
        target_col_start = int(max(0, frame_center[1] - center[1]))
        target_row_end = int(min(self.obs_window[0], target_row_start + drawing_slice.shape[-2]))
        target_col_end = int(min(self.obs_window[1], target_col_start + drawing_slice.shape[-1]))

        frame[:, target_row_start: target_row_end, target_col_start: target_col_end] = drawing_slice
        return frame

    def get_egocentric_observation(self, center: Iterable[float]):
        frame = self.get_frame_at_point(center)
        target_size = (self.obs_resolution[0], self.obs_resolution[1])
        # stupid cv2 needs dimensions in a different order
        frame = image_utils.interpolate(frame, target_size=target_size,
                                        interpolation_method=self.observation_interpolation)
        return frame

    def convert_location_to_pixels(self, location: Iterable[float], origin_bounds: Iterable[Iterable[float]]):
        '''
        If the location is being tracked in a grid, this is converting the location to one for the renderer to use,
        currently in the global resolution
        :param location: Original location being converted
        :param origin_bounds: Original bounds that we will map from, row major, so y then x then for each dimension,
        lower bound first then upper bound
        :return: mapped location from origin bounds to our own global resolution mapping
        '''
        y_scale = (location[0] - origin_bounds[0][0]) / (origin_bounds[0][1] - origin_bounds[0][0])
        x_scale = (location[1] - origin_bounds[1][0]) / (origin_bounds[1][1] - origin_bounds[1][0])
        mapped_y = y_scale * self.global_resolution[0]
        mapped_x = x_scale * self.global_resolution[1]
        return [mapped_y, mapped_x]

    # def convert_distance_to_pixels(self, distance: Iterable[float], origin_bounds: Iterable[Iterable[float]]):
    #     '''
    #     If the location is being tracked in a grid, this is converting the location to one for the renderer to use,
    #     currently in the global resolution
    #     :param location: Original location being converted
    #     :param origin_bounds: Original bounds that we will map from, row major, so y then x then for each dimension,
    #     lower bound first then upper bound
    #     :return: mapped location from origin bounds to our own global resolution mapping
    #     '''
    #     y_scale = (location[0] - origin_bounds[0][0]) / (origin_bounds[0][1] - origin_bounds[0][0])
    #     x_scale = (location[1] - origin_bounds[1][0]) / (origin_bounds[1][1] - origin_bounds[1][0])
    #     mapped_y = y_scale * self.global_resolution[0]
    #     mapped_x = x_scale * self.global_resolution[1]
    #     return [mapped_y, mapped_x]


if __name__ == '__main__':
    renderer = EnvRenderer(resolution=100)
    circle = renderer.draw_circle(point=(1, 2), radius=100)
    print(circle.to())

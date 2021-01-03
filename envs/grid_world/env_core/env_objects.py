from __future__ import annotations

from enum import Enum
from typing import Iterable

import numpy as np

from envs import grid_world as render_shapes


class See(Enum):  # keep nothing as 0
    good_landmark = 0
    bad_landmark = 1
    GoodForeignAgent = 2
    BadForeignAgent = 3


class Actions(Enum):
    up = 0
    right = 1
    down = 2
    left = 3
    stay = 4


DISCRETE_ACTIONS = \
    [
        (-1, 0),  # Up
        (0, 1),  # Right
        (1, 0),  # Down
        (0, -1),  # Left
        (0, 0),  # Stay
        # (-1, 1),  # Up Righte
        # (1, 1),  # Down Right
        # (1, -1),  # Down Left
        # (-1, -1),  # Up Left
    ]


def get_action_unit_vector(index):
    return np.asarray(DISCRETE_ACTIONS[index])


class GridObject(object):
    def __init__(self, id: str, observed_shape: render_shapes.Shape,
                 location: np.ndarray = None, **kwargs):
        self.id = id
        self.location = location
        self.shape = observed_shape

    def __str__(self):
        return '{} {}'.format(self.id, self.__class__)

    def __repr__(self):
        return '{} {}'.format(self.id, self.__class__)

    def place(self, location: Iterable[int]):
        self.location = location

    def get_location(self):
        return self.location


class ActionableItem(GridObject):
    def __init__(self, observed_shape: render_shapes.Shape,
                 id: str, policy=None, location=None, **kwargs):
        super(ActionableItem, self).__init__(observed_shape=observed_shape,
                                             id=id, location=location, **kwargs)
        self.id = id
        self.location = location
        self.policy = policy

    def step(self, obs, **kwargs) -> np.ndarray:
        pass
        # if self.policy is None:
        #     raise NotImplementedError
        # return self.policy.step(obs)

    def interpret_action(self):
        raise NotImplementedError

    def move_towards(self, destination):
        pass

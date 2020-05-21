from __future__ import annotations

from enum import Enum

import numpy as np


class See(Enum):  # keep nothing as 0
    GoodLandmark: 0
    BadLandmark: 1
    GoodForeignAgent: 2
    BadForeignAgent: 3


class Actions(Enum):
    Up: 0
    Right: 1
    Down: 2
    Left: 3
    Stay: 4


ACTIONS = \
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


class GridObject():
    def __init__(self, id: str, location=None):
        self.id = id
        self.location = location

    def __str__(self):
        return '{} {}'.format(self.id, self.__class__)

    def __repr__(self):
        return '{} {}'.format(self.id, self.__class__)

    def place(self, location):
        self.location = location


class ActionableItem(GridObject):
    def __init__(self, id: str, policy=None, location=None, **kwargs):
        super(ActionableItem, self).__init__(id=id, location=location, **kwargs)
        self.id = id
        self.location = location
        self.policy = policy

    def step(self, obs, **kwargs) -> np.ndarray:
        pass
        # if self.policy is None:
        #     raise NotImplementedError
        # return self.policy.step(obs)

    def move_towards(self, destination):
        pass

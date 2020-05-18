from __future__ import annotations

from enum import Enum

import numpy as np

import gym_life.envs.life_channels as ch


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


class Seeable():
    def __init__(self, id: str, see_value, location=None, world=None):
        self.see_value = see_value
        self.id = id
        self.location = location
        self.world = world

    def __str__(self):
        return '{} {}'.format(self.id, self.__class__)

    def __repr__(self):
        return '{} {}'.format(self.id, self.__class__)

    def place(self, location):
        self.location = location


class Actionable():
    pass


class Movable():
    def __init__(self, id: str, see_value: ch.See, location=None, world=None, actions=None):
        super(Movable, self).__init__(id=id, see_value=see_value)
        self.id = id
        self.location = location
        if actions == None:
            actions = np.zeros()

    def place(self, location):
        self.location = location

    def random_move(self):
        pass

    def move_towards(self, destination):
        pass

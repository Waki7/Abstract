from __future__ import annotations

from enum import Enum

import gym_life.envs.life_channels as ch


class See(Enum):  # keep nothing as 0
    GoodLandmark: 0
    BadLandmark: 1
    GoodForeignAgent: 2
    BadForeignAgent: 3


ACTIONS = \
    [
        (-1, 0),  # Up
        (-1, 1),  # Up Righte
        (0, 1),  # Right
        (1, 1),  # Down Right
        (1, 0),  # Down
        (1, -1),  # Down Left
        (0, -1),  # Left
        (-1, -1),  # Up Left
        (0, 0),  # Stay
    ]


class Seeable():
    def __init__(self, id: str, see_value):
        self.see_value = see_value
        self.id = id

    def __str__(self):
        return '{} {}'.format(self.id, self.__class__)

    def __repr__(self):
        return '{} {}'.format(self.id, self.__class__)


class Movable():
    def __init__(self, id: str, see_value: ch.See, location=None):
        super(Movable, self).__init__(id=id, see_value=see_value)
        self.id = id
        self.location = location

    def place(self, location):
        self.location = location

from gym_grid.env_objects.core_env_objects import *


class Landmark(Seeable):
    def __init__(self, id: str, location = None, **kwargs):
        super(Landmark, self).__init__(id=id, **kwargs)
        self.id = id
        self.location = location

    def place(self, location):
        self.location = location

from gym_grid.env_objects.core_env_objects import *

class Object(Seeable):
    def __init__(self, id: str, see_value: ch.See, location=None):
        super(Object, self).__init__(id=id, see_value=see_value)
        self.id = id
        self.location = location

    def place(self, location):
        self.location = location
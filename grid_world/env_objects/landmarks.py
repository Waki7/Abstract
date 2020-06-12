import grid_world.env_objects.core_env_objects as core


class Landmark(core.GridObject):
    def __init__(self, id: str, location=None, **kwargs):
        super(Landmark, self).__init__(id=id, location=location, **kwargs)
        self.id = id
        self.location = location

    def place(self, location):
        self.location = location

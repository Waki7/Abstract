import grid_world.env_objects.core_env_objects as core


class Landmark(core.GridObject):
    def __init__(self, observed_value: tuple, id: str, location=None, **kwargs):
        super(Landmark, self).__init__(observed_value=observed_value, id=id, location=location, **kwargs)
        self.id = id
        self.location = location

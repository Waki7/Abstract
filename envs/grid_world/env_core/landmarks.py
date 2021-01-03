import envs.grid_world.env_core.env_objects as env_objects


class Landmark(env_objects.GridObject):
    def __init__(self, observed_value: tuple, id: str, location=None, **kwargs):
        super(Landmark, self).__init__(observed_value=observed_value, id=id,
                                       location=location, **kwargs)
        self.id = id
        self.location = location

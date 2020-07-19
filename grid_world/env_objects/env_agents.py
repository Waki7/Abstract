import logging

import grid_world.env_objects.core_env_objects as core
import grid_world.rendering.shapes as render_shapes

class EnvAgent(core.ActionableItem):
    def __init__(self, observed_shape: render_shapes.Shape, id: str, policy=None, location=None, **kwargs):
        super(EnvAgent, self).__init__(observed_shape=observed_shape, policy=policy, id=id, location=location, **kwargs)

    # def get_destination(self, action: np.ndarray):
    #     return

    # def move(self, action: int):
    #     return get_action_direction_tuple(action)


class Enemy(EnvAgent):
    def __init__(self, observed_shape: render_shapes.Shape, policy, id: str, location=None, **kwargs):
        super(Enemy, self).__init__(observed_shape=observed_shape, policy=policy, id=id, location=location, **kwargs)


class Friendly(EnvAgent):
    def __init__(self, observed_shape: render_shapes.Shape, policy, id: str, location=None, **kwargs):
        super(Friendly, self).__init__(observed_shape=observed_shape, policy=policy, id=id, location=location, **kwargs)

    def place(self, location):
        self.location = location

    def reset_to_world(self, world):
        self.world = world
        # if we need to add other assignments here

    def step(self, env_input):
        return self.policy(env_input)

    def say(self):
        raise NotImplementedError

    def feed(self):
        self.world.feed_agent(self)

    def give_water(self):
        self.world.give_water_agent(self)

    def update_state(self, update_map):
        raise NotImplementedError

    def log_summary(self):
        logging.debug('{} in room {}'.format(self.id, self.location))

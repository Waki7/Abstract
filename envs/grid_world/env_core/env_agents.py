import logging

import envs.grid_world.env_core.env_objects as core_env_objects
import envs.grid_world.rendering.draw_shapes as draw_shapes


class EnvAgent(core_env_objects.ActionableItem):
    def __init__(self, observed_shape: draw_shapes.Shape, id: str,
                 policy=None, location=None, **kwargs):
        super(EnvAgent, self).__init__(observed_shape=observed_shape,
                                       policy=policy, id=id, location=location,
                                       **kwargs)

    # def get_destination(self, action: np.ndarray):
    #     return

    # def move(self, action: int):
    #     return get_action_direction_tuple(action)


class Enemy(EnvAgent):
    def __init__(self, observed_shape: draw_shapes.Shape, policy, id: str,
                 location=None, **kwargs):
        super(Enemy, self).__init__(observed_shape=observed_shape,
                                    policy=policy, id=id, location=location,
                                    **kwargs)


class Friendly(EnvAgent):
    def __init__(self, observed_shape: draw_shapes.Shape, policy, id: str,
                 location=None, **kwargs):
        super(Friendly, self).__init__(observed_shape=observed_shape,
                                       policy=policy, id=id, location=location,
                                       **kwargs)

    def place(self, location):
        self.location = location

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

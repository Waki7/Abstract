import logging

from gym_grid.env_objects.core_env_objects import *


class ForeignAgent(ActionableItem):
    def __init__(self, id: str, policy=None, location=None, **kwargs):
        super(ForeignAgent, self).__init__(id=id, location=location, **kwargs)


class ForeignEnemies(ForeignAgent):
    def __init__(self, id: str, policy, location=None, **kwargs):
        super(ForeignEnemies, self).__init__(id=id, location=location, policy=policy, **kwargs)


class ForeignFriendlies(ForeignAgent):
    def __init__(self, id: str, policy, location=None, **kwargs):
        super(ForeignFriendlies, self).__init__(id=id, location=location, **kwargs)

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

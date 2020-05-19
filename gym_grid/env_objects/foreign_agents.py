import logging

from gym_grid.env_objects.core_env_objects import *


class ForeignAgent(Actionable):
    def __init__(self, id: str, policy, location=None, **kwargs):
        super(ForeignAgent, self).__init__(id=id, location=location, **kwargs)
        self.location = location
        self.policy = policy

    def place(self, location):
        self.location = location

    def reset_to_world(self, world):
        self.world = world
        # if we need to add other assignments here

    def step(self, env_input):
        return self.policy(env_input)

    def go_to_room(self, location):
        if self.location is not None:
            self.location.remove_person(self)
        location.add_person(self)
        self.location = location

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

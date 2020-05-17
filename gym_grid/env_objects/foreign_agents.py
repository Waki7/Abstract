from gym_grid.env_objects.core_env_objects import *


class ForeignAgent(Seeable):
    def __init__(self, id: str, world: life_env.LifeEnv, see_value, see_close_value: ch.See):
        super(ForeignAgent, self).__init__(id=id, see_value=see_value)
        self.location = None
        self.world = world
        self.see_close_value = see_close_value

    def randomly_switch_rooms(self):
        n_locations = self.world.locations
        self.go_to_room(np.random.choice([self.world.locations], 1, [1.0 / n_locations] * n_locations)[0])

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


class Location(Seeable):
    def __init__(self, id: str, world: life_env.LifeEnv, see_value: ch.See):
        super(Location, self).__init__(id=id, see_value=see_value)
        self.noises = []
        self.people = []
        self.objects = []
        self.world = world

    def add_person(self, person: Person):
        self.people.append(person)

    def remove_person(self, person: Person):
        self.people.remove(person)

    def add_object(self, object: Object):
        self.objects.append(object)

    def remove_object(self, object: Object):
        self.objects.remove(object)

    def update_state(self, agent_prediction):
        [p.update_state(agent_prediction) for p in self.people]

    def log_summary(self):
        logging.debug(
            'location {} has {} objects, {} people, and {} noises in it'.format(self.id, self.objects, self.people,
                                                                                self.noises))


class Mom(Person):
    def __init__(self, id: str, world: life_env.LifeEnv):
        super(Mom, self).__init__(id=id, world=world,
                                  see_value=ch.See.mom,
                                  see_close_value=ch.See.mom_close)

    def update_state(self, agent_prediction):
        if agent_prediction:
            if agent_prediction == ch.Speak.food:
                self.feed()
            if agent_prediction == ch.Speak.water:
                self.give_water()
        old_location = self.location
        if agent_prediction and agent_prediction == ch.Feel.content and model_utils.true_with_probability(.1):
            self.randomly_switch_rooms()
        if old_location is not self.location and self.world.current_room is old_location:
            self.world.food.put_in_room(old_location)


class Sibling(Person):
    def __init__(self, id: str, world: life_env.LifeEnv):
        super(Sibling, self).__init__(id=id, world=world,
                                      see_value=ch.See.sibling,
                                      see_close_value=ch.See.sibling_close)

    def update_state(self, agent_prediction):
        if agent_prediction and agent_prediction == ch.Speak.food:
            self.feed()
        if agent_prediction and agent_prediction == ch.Feel.content and model_utils.true_with_probability(.2):
            self.randomly_switch_rooms()

import RLTasks.LifeSim.life_channels as ch
import numpy as np
from typing import Union
from Tools.TimeBuffer import TimeBuffer
import RLTasks.config as cfg
import gym
from gym import spaces
import logging


def true_with_probability(p):
    return np.random.choice([True, False], 1, [p, 1 - p])


class LifeEnv(gym.Env):
    def __init__(self):
        '''
        This environment is a continuous task (non episodic)
        '''
        self.t = 0
        self.hunger_level = 0
        self.thirst_level = 0
        self.hunger_threshold = 15
        self.thirst_threshold = 5
        self.agent_history = TimeBuffer(5)  # this is arbitrary for the maximum history tracking length
        self.agent_state_channels = [ch.See, ch.Hear, ch.Feel, ch.Movement]
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(sum([len(list(channel)) for channel in self.agent_state_channels]))
        self.reset()

    def reset(self):
        self.room1 = Location(self, ch.See.room1)
        self.room2 = Location(self, ch.See.room2)
        self.outside = Location(self, ch.See.outisde)
        self.mom = Mom(self)
        # self.sibling = Sibling(self)
        # self.food = Object(ch.See.food, ch.See.food_close)
        self.locations = [self.room1, self.room2, self.outside]
        self.t = 0

        self.current_location = self.room1
        self.mom.go_to_room(self.room1)
        # self.sibling.go_to_room(self.room2)
        # self.food.put_in_room(self.room2)

        return self.get_initial_state()

    def step(self, agent_action):
        """
        Args:
            action (object): an action done by the agent, encoded into its channel

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        agent_feel = []
        agent_action_map = ch.decode_to_enum(agent_action, ch.AGENT_ACTION_CHANNELS)
        self.print_summary(agent_action_map)

        previous_feel = self.state_map[ch.Feel]  # todo... proper history tracking
        if ch.Feel.fed in previous_feel and self.hunger_level > self.hunger_threshold:
            agent_feel = [ch.Feel.happy, ch.Feel.content]
            self.hunger_level = 0
        if ch.Feel.drank in previous_feel and self.thirst_level > self.thirst_threshold:
            agent_feel = [ch.Feel.happy, ch.Feel.content]
            self.thirst_level = 0

        if self.hunger_level > self.hunger_threshold:
            agent_feel.append(ch.Feel.hungry)
        if self.thirst_level > self.thirst_threshold:
            agent_feel.append(ch.Feel.thirsty)

        if len(agent_feel) == 0:
            agent_feel = [ch.Feel.content]

        self.hunger_level += 1
        self.thirst_level += 1

        self.current_reward = self.calc_reward()

        self.state_map = self.initialize_empty_map()

        # send feel to agent state
        self.state_map[ch.Feel] = agent_feel

        # translate from agent's prediction to input for itself in next timestep
        if isinstance(agent_action_map, ch.Speak):
            mapped_hear = ch.speak_to_hear_map.get(agent_action_map.value)
            if mapped_hear:
                self.state_map[ch.Hear].append(ch.Hear(mapped_hear))
        if isinstance(agent_action_map, ch.Movement):
            self.state_map[ch.Movement].append(ch.Movement(agent_action_map))

        # render new state
        self.update_state(agent_action_map)

        # persist things maybe not needed todo....
        pass

        self.agent_history.insert(self.t, self.state_map)

        self.t += 1

        # calculate reward
        self.current_reward = self.calc_reward()

        self.state = ch.encode_from_map(self.state_map, ch.AGENT_STATE_CHANNELS)

        return self.state, self.current_reward, False, {}

    def update_state(self, agent_prediction: Union[ch.Movement, ch.Speak]):
        self.current_location.update_state(agent_prediction)
        self.add_setting_to_state()

    def add_setting_to_state(self):
        # self.current_agent_state_map.extend([noises.hear_value for noises in self.current_location.noises])
        self.state_map[ch.See].append(self.current_location.see_value)
        self.state_map[ch.See].extend([person.see_value for person in self.current_location.people])
        self.state_map[ch.See].extend([object.see_value for object in self.current_location.objects])
        for key in self.state_map.keys():
            self.state_map[key] = list(set(
                self.state_map[key]))  # todo if u care make this less retarded, slow but idk it's fine

    def calc_reward(self):
        # todo vectorize a state and calculate similarity to add some reward to it
        return sum([cfg.adjacent_reward_list[feeling.value] for feeling in self.state_map[ch.Feel]])

    def feed_agent(self, giver):
        self.state_map[ch.Feel].append(ch.Feel.fed)
        self.state_map[ch.See].append(giver.close_see_value)
        self.state_map[ch.See].append(ch.See.food_close)
        if true_with_probability(.5):
            self.state_map[ch.Hear].append(ch.Hear.food)

    def give_water_agent(self, giver):
        self.state_map[ch.Feel].append(ch.Feel.drank)
        self.state_map[ch.See].append(giver.close_see_value)
        self.state_map[ch.See].append(ch.See.water_close)
        if true_with_probability(.5):
            self.state_map[ch.Hear].append(ch.Hear.water)

    def get_initial_state(self):
        return ch.encode_from_map(self.initialize_empty_map(), ch.AGENT_STATE_CHANNELS)

    def initialize_empty_map(self):
        return dict(zip(self.agent_state_channels, [[], [], [], []]))

    def get_current_state(self):
        return self.state

    def print_summary(self, agent_action_map):
        logging.debug('agent prediction : ' + str(agent.agent_prediction))
        logging.debug('\nEnv State for agent at timestep ' + str(self.t) + '\n')

        for channel in self.state:
            logging.debug(str(self.state[channel]) + ', ')
        logging.debug('\n')
        agent.log_predictions()
        self.writer.write('env reward is : ' + str(self.current_reward) + '\n')

class Seeable():
    def __init__(self, see_value):
        self.see_value = see_value


class Object(Seeable):
    def __init__(self, see_value: ch.See, close_see_value: ch.See):
        super(Object, self).__init__(see_value)
        self.close_see_value = close_see_value

    def put_in_room(self, room):
        room.add_object(self)


class Person(Seeable):
    def __init__(self, world: LifeEnv, see_value: ch.See, close_see_value: ch.See):
        super(Person, self).__init__(see_value)
        self.location = None
        self.world = world
        self.close_see_value = close_see_value

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


class Location(Seeable):
    def __init__(self, world: LifeEnv, see_value: ch.See):
        super(Location, self).__init__(see_value)
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


class Mom(Person):
    def __init__(self, world: LifeEnv):
        super(Mom, self).__init__(world, ch.See.mom, ch.See.mom_close)

    def update_state(self, agent_prediction):
        if agent_prediction:
            if agent_prediction == ch.Speak.food:
                self.feed()
            if agent_prediction == ch.Speak.water:
                self.give_water()
        old_location = self.location
        if agent_prediction and agent_prediction == ch.Feel.content and true_with_probability(.1):
            self.randomly_switch_rooms()
        if old_location is not self.location and self.world.current_room is old_location:
            self.world.food.put_in_room(old_location)


class Sibling(Person):
    def __init__(self, world: LifeEnv):
        super(Sibling, self).__init__(world, ch.See.sibling, ch.See.sibling_close)

    def update_state(self, agent_prediction):
        if agent_prediction and agent_prediction == ch.Speak.food:
            self.feed()
        if agent_prediction and agent_prediction == ch.Feel.content and true_with_probability(.2):
            self.randomly_switch_rooms()

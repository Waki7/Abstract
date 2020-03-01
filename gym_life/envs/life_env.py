import logging
from typing import Union

import gym
from gym import spaces

import gym_life.envs.life_channels as ch
import gym_life.world.objects as objects
import utils.model_utils as model_utils
from utils.TimeBuffer import TimeBuffer


class LifeEnv(gym.Env):
    def __init__(self, cfg):
        '''
        This environment is a continuous task (non episodic)
        '''
        self.cfg = cfg

        # ---------------------------------------------------------------------------
        # set parameters from config
        # ---------------------------------------------------------------------------
        self.aux_rewards = self.cfg.get('aux_rewards', ch.REWARDS_FEELS)
        self.n_agents = self.cfg.get('n_agents', 1)

        # ---------------------------------------------------------------------------
        # initializing agents according to arbitrary naming scheme
        # ---------------------------------------------------------------------------
        # self.agent_keys = ['agent_{}'.format(i) for i in range(self.n_agents)]

        self.is_episodic = False
        self.hunger_threshold = 15
        self.thirst_threshold = 5
        self.agent_history = TimeBuffer(5)  # this is arbitrary for the maximum history tracking length
        self.agent_state_channels = ch.AGENT_STATE_CHANNELS
        self.agent_action_channels = ch.AGENT_ACTION_CHANNELS
        self.action_space = spaces.Discrete(sum([len(list(channel)) for channel in self.agent_state_channels]))
        self.observation_space = spaces.Discrete(sum([len(list(channel)) for channel in self.agent_state_channels]))
        logging.info('total of {} actions available'.format(self.action_space.n))
        logging.info('total of {} observable discrete observations'.format(self.observation_space.n))

        # ---------------------------------------------------------------------------
        # initializations
        # ---------------------------------------------------------------------------
        self.room1 = None
        self.room2 = None
        self.outside = None
        self.mom = None
        self.locations = None
        self.current_location = None
        self.state_map = None
        self.state = None

        # ---------------------------------------------------------------------------
        # episodic initializations
        # ---------------------------------------------------------------------------
        self.agent_action_map = None
        self.t = 0
        self.hunger_level = 0
        self.thirst_level = 0
        self.current_reward = 0
        self.reset()

    def reset(self):
        self.room1 = objects.Location(id='room1', world=self, see_value=ch.See.room1)
        self.room2 = objects.Location(id='room2', world=self, see_value=ch.See.room2)
        self.outside = objects.Location(id='outside', world=self, see_value=ch.See.outisde)
        self.mom = objects.Mom(id='mom', world=self)
        self.objects = {
            self.room1.id: self.room1,
            self.room2.id: self.room2,
            self.outside.id: self.outside,
            self.mom.id: self.mom,
        }
        # self.sibling = Sibling(self)
        # self.food = Object(ch.See.food, ch.See.food_close)
        self.locations = [self.room1, self.room2, self.outside]
        self.t = 0
        self.hunger_level = 0
        self.thirst_level = 0
        self.current_reward = 0
        self.current_location = self.room1
        self.mom.go_to_room(self.room1)

        # self.sibling.go_to_room(self.room2)
        # self.food.put_in_room(self.room2)
        self.state_map = self.initialize_empty_map()
        self.state = ch.encode_from_map(self.state_map, ch.AGENT_STATE_CHANNELS)

        return self.state

    def add_agent(self):
        pass

    def step(self, agent_actions):
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
        agent_action_map = ch.decode_to_enum(agent_actions, ch.AGENT_ACTION_CHANNELS)
        self.agent_action_map = agent_action_map

        self.log_summary(agent_action_map)

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
        return sum([self.aux_rewards[feeling.value] for feeling in self.state_map[ch.Feel]])

    def feed_agent(self, giver):
        self.state_map[ch.Feel].append(ch.Feel.fed)
        self.state_map[ch.See].append(giver.see_close_value)
        self.state_map[ch.See].append(ch.See.food_close)
        if model_utils.true_with_probability(.5):
            self.state_map[ch.Hear].append(ch.Hear.food)

    def give_water_agent(self, giver):
        self.state_map[ch.Feel].append(ch.Feel.drank)
        self.state_map[ch.See].append(giver.see_close_value)
        self.state_map[ch.See].append(ch.See.water_close)
        if model_utils.true_with_probability(.5):
            self.state_map[ch.Hear].append(ch.Hear.water)

    def initialize_empty_map(self):
        return dict(zip(self.agent_state_channels, [[], [], [], []]))

    def get_current_state(self):
        return self.state

    def log_summary(self):
        logging.debug('\n___________start step {}_______________'.format(self.t))

        if self.agent_action_map:
            logging.debug('agent\' latest prediction : ' + str(self.agent_action_map))
        logging.debug('Env State for agent at timestep ' + str(self.t))
        for object_id in self.objects.keys():
            self.objects[object_id].log_summary()
        for channel in self.state_map:
            val = self.state_map[channel]
            if len(val) > 0:
                logging.debug('channel {}: {}, '.format(channel, str(self.state_map[channel])))
        # agent.log_predictions()
        logging.debug('env reward is : ' + str(self.current_reward))
        logging.debug('___________end step {}_______________\n'.format(self.t))

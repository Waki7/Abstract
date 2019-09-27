import RLTasks.config as cfg
import torch as torch
import numpy as np
from torch.autograd import Variable

class AgentInterface:
    def __init__(self):
        self.env_in_size = NotImplemented
        self.hidden_in_size = NotImplemented
        self.in_size = NotImplemented
        self.is_episodic = False
        self.out_size = NotImplemented
        self.t = 0
        self.input_is_discrete = True
        self.outputIsDiscrete = True
        self.exploitation_penalty = cfg.exploitation_penalty

    def step(self, env_input, env_reward, episode_done = False):
        raise NotImplementedError


    def get_action(self, state):
        raise NotImplementedError

    def get_pred_indeces(self, predictions):
        raise NotImplementedError

    def calc_rewards(self):
        raise NotImplementedError

    def episodeDone(self):
        return True #override if the task is episodic (only return true for end of task)
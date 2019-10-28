from agent_algorithms.cra_agent import *
import logging
import torch.utils.tensorboard
from networks.base_networks import *
import agent_algorithms
import config
from shutil import copy as copy_file
import yaml
import numpy as np
import os
import random
import torch
import config as cfg

def train():
    env =



def main():
    name = 'CartPole-v0'  #
    # name = 'Life-v0'
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    env = gym.make(name)
    with open('config.yaml') as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)
    network = None # default fallback
    agent = agent_algorithms.A2CAgent(env=env, model=network)
    teach_agents(env=env, agent=agent)


if __name__ == "__main__":
    main()

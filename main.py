from agent_algorithms.cra_agent import *
import logging

from networks.base_networks import *
import agent_algorithms

from shutil import copy as copy_file

import numpy as np
import os
import random
import torch
import config as cfg



def main():
    name = 'CartPole-v0'  #
    # name = 'Life-v0'
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    env = gym.make(name)

    network = None # default fallback
    agent = agent_algorithms.A2CAgent(env=env, model=network)
    teach_agents(env=env, agent=agent)


if __name__ == "__main__":
    main()

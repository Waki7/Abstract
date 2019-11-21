from agent_algorithms.cra_agent import *
import logging
import tensorboard
from agent_controllers.factory import CONTROLLER_REGISTERY
from networks.base_networks import *
import agent_algorithms
import config
from shutil import copy as copy_file
import yaml
import numpy as np
import os
import random
import torch

with open('config.yaml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)


def train(algorithm, env_namespace):
    env_cfg = cfg['env'][env_namespace]
    algorithm_cfg = cfg['agents'][algorithm]
    trainer = CONTROLLER_REGISTERY[algorithm_cfg['controller_name']](env_cfg, algorithm_cfg)
    trainer.teach_agents(cfg['training'])


def main():
    logging.basicConfig(level=
                        logging.INFO
                        # logging.DEBUG
                        )
    # train('a2c', 'life')
    train('a2c', 'cart')
    train('pg', 'cart')

if __name__ == "__main__":
    main()

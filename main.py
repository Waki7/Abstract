from agent_controllers.factory import CONTROLLER_REGISTERY
from networks.base_networks import *
import yaml
import gym_life
import grid_world

with open('config.yaml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)


def train(algorithm, env_namespace):
    env_cfg = cfg['env'][env_namespace]
    algorithm_cfg = cfg['agents'][algorithm]
    trainer = CONTROLLER_REGISTERY[algorithm_cfg['controller_name']](env_cfg, algorithm_cfg)
    trainer.teach_agents(cfg['training'])


def main():
    logging.basicConfig(level=
                        # logging.INFO
                        logging.DEBUG
                        )
    # train('a2c', 'cart')
    # train('social', 'cart')
    train('grid', 'grid')
    # train('exp', 'cart')
    # train('ccra', 'cart')
    # train('cra', 'cart')
    # train('a2c', 'life')
    # train('a2c', 'beam')
    # train('pg', 'cart')

if __name__ == "__main__":
    main()

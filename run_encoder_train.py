import logging

import yaml


import grid_world.env_factory as env_factory
import grid_world.encoder_trainers as enc_trainers
from networks.net_factory import get_network
from agent_controllers.factory import CONTROLLER_REGISTERY
import utils.model_utils as model_utils

with open('./configs/cfg_execution.yaml') as f:
    CFG_EXECUTION = yaml.load(f, Loader=yaml.FullLoader)

with open('./configs/cfg_env.yaml') as f:
    CFG_ENV = yaml.load(f, Loader=yaml.FullLoader)

with open('./configs/cfg_agent.yaml') as f:
    CFG_AGENT = yaml.load(f, Loader=yaml.FullLoader)

with open('./configs/cfg_network.yaml') as f:
    CFG_NETWORK = yaml.load(f, Loader=yaml.FullLoader)


def train_state_encoder(network, env_namespace):
    env_cfg = CFG_ENV[env_namespace]

    trainer: enc_trainers.StateEncodingProtocol = env_factory.get_state_trainer(env_cfg)
    obs_space = trainer.get_in_spaces()
    out_space = trainer.get_out_spaces()
    in_shapes = model_utils.spaces_to_shapes(obs_space)
    out_shapes = model_utils.spaces_to_shapes(out_space)
    get_network()

# def func2(a, b, c=1):
#     print('funct2, {}'.format(a))
#     print('funct2, {}'.format(b))
#     print('funct2, {}'.format(c))
#
#

# def func(a, *method_args, **kwargs):
#     print(a)
#     print(method_args)
#     print(kwargs)
#     func2(*method_args, **kwargs)
#

def main():
    # func('a', ('met', 'args'), c='keyword')
    # print(exit(9))
    logging.basicConfig(level=
                        # logging.INFO
                        logging.DEBUG
                        )
    train_state_encoder('a2c', 'grid')


if __name__ == "__main__":
    main()

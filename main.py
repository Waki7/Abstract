import yaml

from agent_controllers.factory import CONTROLLER_REGISTERY
from networks.base_networks import *

with open('./configs/cfg_execution.yaml') as f:
    CFG_EXECUTION = yaml.load(f, Loader=yaml.FullLoader)

with open('./configs/cfg_env.yaml') as f:
    CFG_ENV = yaml.load(f, Loader=yaml.FullLoader)

with open('./configs/cfg_agent.yaml') as f:
    CFG_AGENT = yaml.load(f, Loader=yaml.FullLoader)

with open('./configs/cfg_network.yaml') as f:
    CFG_NETWORK = yaml.load(f, Loader=yaml.FullLoader)


def train(algorithm, env_namespace):
    env_cfg = CFG_ENV[env_namespace]
    algorithm_cfg = CFG_AGENT[algorithm]
    trainer = CONTROLLER_REGISTERY[algorithm_cfg['controller_name']](env_cfg, algorithm_cfg)
    trainer.teach_agents(CFG_EXECUTION['training'])


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
    train('a2c', 'grid')


if __name__ == "__main__":
    main()

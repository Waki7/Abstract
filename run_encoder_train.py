import logging

import yaml

from controllers.visual.encoder_trainer import EnvTraining

with open('./configs/cfg_execution.yaml') as f:
    CFG_EXECUTION = yaml.load(f, Loader=yaml.FullLoader)

with open('./configs/cfg_env.yaml') as f:
    CFG_ENV = yaml.load(f, Loader=yaml.FullLoader)

with open('./configs/cfg_agent.yaml') as f:
    CFG_AGENT = yaml.load(f, Loader=yaml.FullLoader)

with open('./configs/cfg_network.yaml') as f:
    CFG_NETWORK = yaml.load(f, Loader=yaml.FullLoader)


def train_state_encoder(network_namespace, env_namespace):
    network_cfg = CFG_NETWORK[network_namespace]
    env_cfg = CFG_ENV[env_namespace]
    trainer: EnvTraining = EnvTraining(env_cfg=env_cfg, network_cfg=network_cfg)
    trainer.train(CFG_EXECUTION['encoder_training'])


def main():
    # func('a', ('met', 'args'), c='keyword')
    # print(exit(9))
    logging.basicConfig(level=
                        # logging.INFO
                        logging.DEBUG
                        )
    train_state_encoder('mobile', 'grid')


if __name__ == "__main__":
    main()

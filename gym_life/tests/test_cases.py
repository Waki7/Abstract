import logging

import gym
import numpy as np
import yaml

import gym_life.envs.life_channels as ch


class ManualAgent():
    # this agent can work with environments x, y, z (life and gym envs)
    # try to make the encoding part separate
    def __init__(self):
        pass

    def step(self, env_input):
        print(env_input)


def vector_to_enum():
    with open('config.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    life_env = gym.make(cfg['env']['life']['name'], cfg=cfg)
    action_channels = ch.AGENT_ACTION_CHANNELS
    print(action_channels)
    action = np.zeros((life_env.action_space.n))
    action[2] = 1.
    action_map = ch.decode_to_enum(action, action_channels)
    print(action_map)


def enum_map_to_vector():
    channels = [ch.Speak, ch.Feel]
    values = [
        [ch.Speak.water, ch.Speak.nothing, ],
        [ch.Feel.content, ],
    ]
    action_map = dict(zip(channels, values))
    vector = ch.encode_from_map(map=action_map, channels=channels)
    print(vector)


def case_1():
    with open('config.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    life_env = gym.make(cfg['env']['life']['name'], cfg=cfg)

    state = life_env.reset()

    print(state)

    vector = ch.agent_action_vector([ch.Speak.water])

    life_env.step(vector)
    life_env.log_summary()
    print(state)
    print(exit(9))


def interactive():
    pass


def main():
    logging.basicConfig(level=
                        # logging.INFO
                        logging.DEBUG
                        )
    # enum_map_to_vector()
    case_1()
    interactive()


if __name__ == "__main__":
    main()

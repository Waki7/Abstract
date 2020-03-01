import gym
import yaml
import gym_life
import logging

class ManualAgent():
    # this agent can work with environments x, y, z (life and gym envs)
    # try to make the encoding part separate
    def __init__(self):
        pass

    def step(self, env_input):
        print(env_input)

def case_1():
    with open('config.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    life_env = gym.make(cfg['env']['life']['name'], cfg=cfg)
    state = life_env.reset()
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
    case_1()
    interactive()

if __name__ == "__main__":
    main()
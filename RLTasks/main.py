from RLTasks.network_controllers.cra_agent import *
from RLTasks.LifeSim.life_simulation import *
import logging

from RLTasks.network_controllers.life_network import *
from RLTasks.network_controllers.gym_network import *
from RLTasks.network_controllers.pg_agent import PGAgent

from RLTasks.openai_gym.gym_simulation import *

from shutil import copy as copy_file

import numpy as np
import os
import random
import torch
import RLTasks.config as cfg


# import pybulletgym
# import gym

def start_experiment():
    np.random.seed(23)
    random.seed(23)
    torch.manual_seed(99)
    if not os.path.exists(cfg.results_path):
        os.makedirs(cfg.results_path)
    experiment_writer = open(cfg.results_path + 'simulation.txt', 'w+')
    copy_file(cfg.__file__, cfg.results_path)
    return experiment_writer


def teach_agents(env: gym.Env, agent):
    is_episodic = not hasattr(env, 'is_episodic') or (hasattr(env, 'is_episodic') and env.is_episodic)
    update_rate = 1 if is_episodic else cfg.experiment.UPDATE_RATE
    max_episodes = cfg.experiment.MAX_EPISODES if is_episodic else 1
    all_rewards = []
    numsteps = []
    avg_numsteps = []

    for episode in range(max_episodes):
        state = env.reset()
        rewards = []
        step = 0
        while True:
            action = agent.step(state)
            state, reward, episode_end, _ = env.step(action)
            rewards.append(reward)
            agent.update_policy(reward, episode_end)
            if (is_episodic and episode_end) or (not is_episodic and step % update_rate == 0):
                numsteps.append(step)
                avg_numsteps.append(np.mean(numsteps[-cfg.experiment.EVAL_REWARDS_WINDOW:]))
                all_rewards.append(np.sum(rewards))
                logging.debug(
                    "episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode, np.round(
                        np.sum(rewards), decimals=3), np.round(
                        np.mean(all_rewards[-cfg.experiment.EVAL_REWARDS_WINDOW:]), decimals=3), step))
                if is_episodic:
                    break
                else:
                    rewards = []
            if step > cfg.experiment.MAX_STEPS:
                break
            step += 1

    if cfg.results_path is not '':
        plot_rewards = [np.mean(all_rewards[i:i+50]) for i in range(0, len(all_rewards), 50)]
        print(plot_rewards)
        print(cfg.results_path)
        plt.plot(plot_rewards)
        plt.savefig(cfg.results_path + 'averageRewards.png')


def LifeSim():
    experiment_writer = start_experiment()
    network = LifeNetwork().cuda()
    simulation = LifeSimulation(writer=experiment_writer)
    agent = CRAgent(network, simulation.env)
    simulation.teach_agents(agent)
    agent.plot_results()


def GymSim():
    experiment_writer = start_experiment()
    simulation = GymSimulation(writer=experiment_writer)

    network = PolicyNetwork(simulation.env)
    agent = PGAgent(network, simulation.env)

    simulation.teach_agents(agent)
    agent.plot_results()


def main():
    # GymSim()
    # name = 'CartPole-v0' #
    name = 'Life-v0'
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    env = gym.make(name)

    network = PolicyNetwork(env)
    agent = PGAgent(model=network, env=env)
    teach_agents(env=env, agent=agent)


if __name__ == "__main__":
    main()

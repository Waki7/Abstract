from agents.cra_agent import *
import logging

from networks.base_networks import *
from networks import A2CAgent

from shutil import copy as copy_file

import numpy as np
import os
import random
import torch
import config as cfg


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

    plot_rewards = [np.mean(all_rewards[i:i + 50]) for i in range(0, len(all_rewards), 50)]
    env_id = (env.unwrapped.spec.id if not hasattr(env, 'envs') else env.envs[0].unwrapped.spec.id)
    results_path = '{}{}{}/{}'.format(cfg.paths.RESULTS, agent.__class__.__name__, cfg.experiment.VARIATION, env_id)
    print(plot_rewards)
    print(results_path)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    plt.plot(plot_rewards)
    print(env)
    plt.savefig('{}/averageRewards.png'.format(results_path))


def main():
    name = 'CartPole-v0'  #
    # name = 'Life-v0'
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    env = gym.make(name)

    network = None # default fallback
    agent = A2CAgent(env=env, model=network)
    teach_agents(env=env, agent=agent)


if __name__ == "__main__":
    main()

import gym
import sys
import numpy as np
from RLTasks.network_controllers.pg_agent import PGAgent
import RLTasks.config as cfg


class GymSimulation():
    def __init__(self, writer=sys.stdout):
        self.env = gym.make(cfg.gym.gym_env)
        self.t = 0
        self.currentReward = 0
        self.max_steps = 10000
        self.writer = writer

    def teach_agents(self, agent: PGAgent):
        self.state = self.env.state
        all_rewards = []
        numsteps = []
        avg_numsteps = []

        for episode in range(cfg.MAX_EPISODES):
            self.env.render()

            rewards = []
            for step in range(cfg.max_steps):
                action = agent.step(state)
                state, reward, episode_end, _ = self.env.step(action)
                rewards.append(reward)
                agent.update_policy(reward, episode_end)

                if episode_end:
                    numsteps.append(step)
                    avg_numsteps.append(np.mean(numsteps[-10:]))
                    all_rewards.append(np.sum(rewards))
                    if episode % cfg.episode_eval_window == 0:
                        sys.stdout.write(
                            "episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode, np.round(
                                np.sum(rewards), decimals=3), np.round(np.mean(all_rewards[-10:]), decimals=3), step))
                    break

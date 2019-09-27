import sys
import gym_life.envs.life_env as world
import RLTasks.config as cfg

class LifeSimulation():
    def __init__(self, activeInteraction=False,
                 preTrainIterations=100, writer=sys.stdout):
        self.t = 0
        self.env = world.LifeEnv()
        self.agent_prediction = None
        self.writer = writer

        self.current_reward = 0

    def teach_agents(self, agent):
        self.state = self.env.get_initial_state()
        for i in range(cfg.iterations):
            agent_prediction = agent.step(self.state, self.current_reward)
            self.print_summary()
            self.state, reward = self.env.step(agent_prediction)
            agent.update_policy(reward)

            self.writer.write('end of timestep ' + str(self.t) + '_____________________________________\n\n')
            print('end of timestep ', self.t, '_____________________________________\n\n')
            self.store_state()
            self.t += 1



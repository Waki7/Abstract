from agent_algorithms.factory import register_agent
import sys
import matplotlib.pyplot as plt
from utils.TimeBuffer import TimeBuffer
from networks.base_networks import *

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
type = torch.float
args = {'device': device, 'dtype': type}

@register_agent
class PGAgent():
    # this agent can work with environments x, y, z (life and gym envs)
    # try to make the encoding part separate
    def __init__(self, env: gym.Env, model=None):
        assert isinstance(env, gym.Env)
        assert isinstance(env.action_space, gym.spaces.Discrete)

        self.model = model if model else ActorFCNetwork(env)
        self.is_episodic = not hasattr(env, 'is_episodic') or (hasattr(env, 'is_episodic') and env.is_episodic)
        self.policy_net = model
        self.reward = 0
        self.testing_rewards = TimeBuffer(cfg.rewards_eval_window)
        self.average_rewards = []
        self.probs = []
        self.rewards = []
        self.t = 0
        self.optimizer = getattr(torch.optim, cfg.gym.OPTIMIZER)(self.model.parameters(), lr=cfg.gym.LR)

    def step(self, env_input):
        action, prob = self.policy_net.get_action(env_input)
        self.probs.append(prob)
        self.t += 1
        return action

    def update_policy(self, env_reward, episode_end):
        self.rewards.append(env_reward)
        if episode_end or (not self.is_episodic and self.t == cfg.pg.CONTINUOUS_EPISODE_LENGTH):
            discounted_rewards = [0]
            while self.rewards:
                # latest reward + (future reward * gamma)
                discounted_rewards.insert(0, self.rewards.pop() + (cfg.discount_factor * discounted_rewards[0]))
            discounted_rewards.pop(-1)  # remove the extra 0 placed before the loop

            discounted_rewards = torch.tensor(discounted_rewards)
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                    discounted_rewards.std() + 1e-9)  # normalize discounted rewards

            policy_gradient = []
            for prob, Gt in zip(self.probs, discounted_rewards):
                policy_gradient.append(-torch.log(prob) * Gt)

            self.optimizer.zero_grad()
            policy_gradient = torch.stack(policy_gradient).sum()
            policy_gradient.backward()
            self.optimizer.step()

            self.t = 0
            self.rewards = []
            self.probs = []

    def log_predictions(self, writer=sys.stdout):
        writer.write('\nAgent Summary at timestep ' + str(self.t) + '\n')
        writer.write('prediction to environment: ' + str(self.model.get_env_pred_val()) + '\n')
        writer.write(str(self.pred_val) + ', ' + str(self.pred_feel_val))
        writer.write('\n\n full reward is: ' + str(self.reward))
        writer.write('\n')
        writer.flush()

    def store_results(self, reward):
        self.testing_rewards.insert(self.t, reward)
        if not self.t % cfg.rewards_eval_window:
            self.average_rewards.append(np.average(self.testing_rewards.getData()))

    def plot_results(self):
        if cfg.results_path is not None:
            print('trying')
            print(self.average_rewards)
            print(cfg.results_path)
            plt.plot(self.average_rewards)
            plt.savefig(cfg.results_path + 'averageRewards.png')

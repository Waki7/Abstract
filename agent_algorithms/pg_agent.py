from agent_algorithms.factory import register_agent
import logging
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
    def __init__(self, is_episodic, cfg, actor):
        self.actor = actor
        self.n_actions = self.actor.n_actions

        ##########################################################################################
        # set cfg parameters
        ##########################################################################################
        self.update_threshold = cfg.get('update_threshold', -1)
        self.td_step = cfg.get('td_step', -1)
        self.discount_factor = cfg.get('discount_factor', settings.defaults.DISCOUNT_FACTOR)
        self.entropy_coef = cfg.get('entropy_coef', settings.defaults.ENTROPY_COEF)
        logging.debug(' update_threshold : ', self.update_threshold)
        logging.debug(' td_step : ', self.td_step)
        logging.debug(' disount_factor : ', self.discount_factor, '\n')

        self.is_episodic = is_episodic
        self.reward = 0
        self.action_probs = []
        self.probs = []
        self.rewards = []
        self.t = 0

    def step(self, env_input):
        env_input = torch.from_numpy(env_input).to(settings.DEVICE).float().unsqueeze(0)
        probs = self.actor.forward(env_input).squeeze(0)

        self.probs.append(probs)

        action = np.random.choice(self.n_actions, p=probs.detach().cpu().numpy())
        self.action_probs.append(probs[action])

        self.t += 1
        return action

    def update_policy(self, env_reward, episode_end, new_state=None):
        ret_loss = 0
        self.rewards.append(env_reward)
        should_update = self.should_update(episode_end, env_reward)
        if should_update:
            discounted_rewards = [0]
            while self.rewards:
                # latest reward + (future reward * gamma)
                discounted_rewards.insert(0, self.rewards.pop(-1) + (self.discount_factor * discounted_rewards[0]))
            discounted_rewards.pop(-1)  # remove the extra 0 placed before the loop

            Q_val = torch.tensor(discounted_rewards).to(**args)
            Q_val = (Q_val - Q_val.mean()) / (Q_val.std() + 1e-9)

            action_probs = torch.stack(self.action_probs)

            action_log_prob = torch.log(action_probs)
            actor_loss = (-action_log_prob * Q_val).sum()
            actor_loss.backward()
            ret_loss = actor_loss.detach().cpu().item()

            self.update_networks()

        if should_update:
            self.reset_buffers()
        return ret_loss

    def update_networks(self):
        self.actor.update_parameters()

    def reset_buffers(self):
        self.rewards = []
        self.probs = []
        self.action_probs = []

    def should_update(self, episode_end, reward):
        steps_since_update = len(self.rewards) + 1
        td_update = self.td_step != -1 and steps_since_update % self.td_step == 0
        if self.update_threshold == -1:  # not trying the threshold updater
            return episode_end or td_update
        return episode_end or reward >= self.update_threshold


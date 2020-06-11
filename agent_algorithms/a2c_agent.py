from agent_algorithms.factory import register_agent
from networks.base_networks import *

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
type = torch.float
args = {'device': device, 'dtype': type}


@register_agent
class A2CAgent():
    # this agent can work with environments x, y, z (life and gym envs)
    # try to make the encoding part separate
    def __init__(self, is_episodic, cfg, actor, critic=None):
        self.is_ac_shared = critic is None
        self.ac = None
        if self.is_ac_shared:
            self.ac = actor
            self.n_actions = self.ac.n_actions
        else:
            self.actor = actor
            self.n_actions = self.actor.n_actions
            self.critic = critic

        ##########################################################################################
        # set cfg parameters
        ##########################################################################################
        self.update_threshold = cfg.get('update_threshold', -1)
        self.td_step = cfg.get('td_step', -1)
        self.discount_factor = cfg.get('discount_factor', settings.defaults.DISCOUNT_FACTOR)
        self.entropy_coef = cfg.get('entropy_coef', settings.defaults.ENTROPY_COEF)
        self.supervised_loss = cfg.get('supervised_loss', False)
        logging.debug(' update_threshold : {}'.format(self.update_threshold))
        logging.debug(' td_step : {}'.format(self.td_step))
        logging.debug(' discount_factor : {}'.format(self.discount_factor))
        logging.debug(' entropy_coef : {}'.format(self.entropy_coef))

        self.is_episodic = is_episodic
        self.reward = 0

        self.batch_actions = []
        self.batch_probs_selected = []
        self.batch_probs = []
        self.batch_value_estimates = []
        self.rewards = []
        self.t = 0

    def get_action(self):
        if len(self.batch_actions) == 0:
            return None
        return self.batch_actions[-1]

    def step(self, env_input):
        if self.ac is not None:
            probs, estimates = self.ac.forward(env_input)
        else:
            probs = self.actor.forward(env_input)
            estimates = self.critic.forward(env_input)

        batch_actions = model_utils.random_choice_prob_batch(self.n_actions,
                                                             probs.detach().cpu().numpy())
        selected_probs = torch.stack([probs[i][action] for i, action in enumerate(batch_actions)])

        self.batch_actions.append(batch_actions)
        self.batch_probs_selected.append(selected_probs)
        self.batch_probs.append(probs)
        self.batch_value_estimates.append(estimates.squeeze(-1))

        self.t += 1
        return self.batch_actions[-1]

    def update_policy(self, env_reward, episode_end, new_state=None):
        ret_loss = {}
        self.rewards.append(env_reward)
        should_update = self.should_update(episode_end, env_reward)
        if should_update:
            discounted_rewards = [0]
            while self.rewards:
                # latest reward + (future reward * gamma)
                discounted_rewards.insert(0, self.rewards.pop(-1) + (self.discount_factor * discounted_rewards[0]))
            discounted_rewards.pop(-1)  # remove the extra 0 placed before the loop

            Q_val = torch.tensor(discounted_rewards).to(**args)
            V_estimate = torch.cat(self.batch_value_estimates, dim=0)
            advantage = Q_val - V_estimate
            if advantage.shape[0] > 1:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-9)  # normalizing the advantage

            probs = torch.stack(self.probs)
            action_probs = torch.stack(self.action_probs)

            action_log_prob = torch.log(action_probs)
            actor_loss = (-action_log_prob * advantage.detach()).sum()

            critic_loss = F.smooth_l1_loss(input=V_estimate, target=Q_val,
                                           reduction='mean')  # .5 * advantage.pow(2).mean()

            entropy_loss = (torch.log(probs) * probs).mean()

            ac_loss = actor_loss + critic_loss + (self.entropy_coef * entropy_loss)

            ac_loss.backward()
            ret_loss['actor_loss'] = actor_loss.detach().cpu().item()
            ret_loss['critic_loss'] = critic_loss.detach().cpu().item()

            self.update_networks()
            self.reset_buffers()

        return ret_loss

    def update_networks(self):
        if self.is_ac_shared:
            self.ac.update_parameters()
        else:
            self.actor.update_parameters()
            self.critic.update_parameters()

    def reset_buffers(self):
        self.rewards = []
        self.probs = []
        self.action_probs = []
        self.actions = []
        self.batch_value_estimates = []

    def should_update(self, episode_end, reward):
        steps_since_update = len(self.rewards) + 1
        td_update = self.td_step != -1 and steps_since_update % self.td_step == 0
        if self.update_threshold == -1:  # not trying the threshold updater
            return episode_end or td_update
        update = episode_end or np.abs(reward) >= self.update_threshold
        return update

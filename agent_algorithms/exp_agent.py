import utils.model_utils as model_utils
from agent_algorithms.factory import register_agent
from networks.base_networks import *

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
type = torch.float
args = {'device': device, 'dtype': type}


@register_agent
class ExpAgent():
    # this agent can work with environments x, y, z (life and gym envs)
    # try to make the encoding part separate
    def __init__(self, is_episodic, cfg, actor, critic=None, attention=None):
        self.is_ac_shared = critic is None
        self.ac = None
        if self.is_ac_shared:
            self.ac = actor
            self.n_actions = self.ac.n_actions
        else:
            self.actor = actor
            self.n_actions = self.actor.n_actions
            self.critic = critic
            self.attention = attention

        ##########################################################################################
        # set cfg parameters
        ##########################################################################################
        self.update_threshold = cfg.get('update_threshold', -1)
        self.td_step = cfg.get('td_step', -1)
        self.discount_factor = cfg.get('discount_factor', settings.defaults.DISCOUNT_FACTOR)
        self.entropy_coef = cfg.get('entropy_coef', settings.defaults.ENTROPY_COEF)
        self.supervised_loss = cfg.get('supervised_loss', False)
        self.attention_reward = cfg.get('attention_reward', False)
        logging.debug(' update_threshold : ', self.update_threshold)
        logging.debug(' td_step : ', self.td_step)
        logging.debug(' discount_factor : ', self.discount_factor, '\n')
        logging.debug(' entropy_coef : ', self.entropy_coef, '\n')
        logging.debug(' supervised_loss : ', self.supervised_loss, '\n')
        logging.debug(' attention_reward : ', self.attention_reward, '\n')

        self.is_episodic = is_episodic
        self.reward = 0
        self.action_probs = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.value_estimates = []
        self.t = 0

    def step(self, env_input):
        env_input = model_utils.convert_env_input(env_input)
        if self.ac is not None:
            probs, estimates = self.ac.forward(env_input)
        else:
            probs = self.actor.forward(env_input).squeeze(0)
            estimates = self.critic.forward(env_input).squeeze(0)

        self.probs.append(probs)
        self.value_estimates.append(estimates)

        action = np.random.choice(self.n_actions, p=probs.detach().cpu().numpy())
        self.action_probs.append(probs[action])
        self.actions.append(action)
        self.t += 1
        return action

    def update_policy(self, env_reward, episode_end, new_state=None):
        ret_loss = 0
        self.rewards.append(env_reward)
        should_update = self.should_update(episode_end, env_reward)
        if should_update:

            V_estimate = torch.cat(self.value_estimates, dim=0)

            if self.attention_reward:
                discounted_rewards = []
                for i in range(0, len(self.rewards)):
                    rewards = self.rewards if i == 0 else self.rewards[:-i]
                    attention_input = torch.tensor(rewards).to(settings.DEVICE)
                    weighted_reward = self.attention(attention_input).sum()
                    discounted_rewards.insert(0, weighted_reward)
                discounted_rewards = torch.tensor(discounted_rewards).to(**args)
                # if len(discounted_rewards) > 1:
                #     discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / \
                #                          (discounted_rewards.std() + 1e-9)  # normalizing the advantage
                advantage = discounted_rewards

            else:
                discounted_rewards = [0]
                while self.rewards:
                    # latest reward + (future reward * gamma)
                    discounted_rewards.insert(0, self.rewards.pop(-1) + (self.discount_factor * discounted_rewards[0]))
                discounted_rewards.pop(-1)  # remove the extra 0 placed before the loop
                discounted_rewards = torch.tensor(discounted_rewards).to(**args)
                if discounted_rewards.shape[0] > 1:
                    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / \
                                         (discounted_rewards.std() + 1e-9)  # normalizing the advantage
                advantage = discounted_rewards - V_estimate
                advantage = advantage#self.attention(advantage.detach())
                # print(advantage)
                # print(attention_advantage)
            probs = torch.stack(self.probs)
            action_probs = torch.stack(self.action_probs)

            action_log_prob = torch.log(action_probs)
            if self.supervised_loss:
                target_actions = model_utils.get_target_action(self.n_actions, self.actions, advantage)
                actor_loss = nn.MSELoss()(input=probs, target=target_actions)
                actor_loss *= advantage.detach().sum()
            else:
                actor_loss = (-action_log_prob * advantage.detach()).sum()

            critic_loss = F.smooth_l1_loss(input=V_estimate, target=discounted_rewards,
                                           reduction='mean')  # .5 * advantage.pow(2).mean()
            entropy_loss = (torch.log(probs) * probs).mean()

            ac_loss = actor_loss + critic_loss + (self.entropy_coef * entropy_loss)

            ac_loss.backward()
            ret_loss = ac_loss.detach().cpu().item()

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
        self.value_estimates = []

    def should_update(self, episode_end, reward):
        steps_since_update = len(self.rewards) + 1
        td_update = self.td_step != -1 and steps_since_update % self.td_step == 0
        if self.update_threshold == -1:  # not trying the threshold updater
            return episode_end or td_update
        return episode_end or reward >= self.update_threshold

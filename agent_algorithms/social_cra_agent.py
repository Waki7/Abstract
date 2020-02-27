import logging
import sys

import numpy as np
import torch
import torch.nn.functional as F

import settings
import utils.model_utils as model_utils
from agent_algorithms.factory import register_agent


@register_agent
class SocialCRAAgent():
    # this agent can work with environments x, y, z (life and gym envs)
    # todo move cragent controller here, and move this stuff in life network
    # try to make the encoding part separate
    def __init__(self, is_episodic, cfg, ac):
        self.ac = None
        self.cfg = cfg

        self.ac = ac
        self.n_actions = self.ac.n_actions
        self.reward = 0
        self.imagination_decay = .001

        ##########################################################################################
        # episodic initializations
        ##########################################################################################
        self.inputs = []
        self.rewards = []
        self.value_estimates = []
        self.action_probs = []
        self.action_taken_probs = []

        self.concept_probs = []
        self.encoder_estimates = []
        self.imagine_probs = []
        self.imagine_taken_probs = []
        self.internal_estimates = []

        self.action = torch.zeros((1, self.n_actions)).to(settings.DEVICE)
        self.inputs = []
        self.t = 0

        ##########################################################################################
        # set cfg parameters
        ##########################################################################################
        self.update_threshold = cfg.get('update_threshold', -1)
        self.td_step = cfg.get('td_step', -1)
        self.discount_factor = cfg.get('discount_factor', settings.defaults.DISCOUNT_FACTOR)
        self.entropy_coef = cfg.get('entropy_coef', settings.defaults.ENTROPY_COEF)
        self.entropy_coef = cfg.get('entropy_coef', settings.defaults.ENTROPY_COEF)
        self.n_concepts = cfg.get("n_concepts", 32)

        logging.debug(' update_threshold : {}'.format(self.update_threshold))
        logging.debug(' td_step : {}'.format(self.td_step))
        logging.debug(' discount_factor : {}'.format(self.discount_factor))
        logging.debug(' entropy_coef : {}'.format(self.entropy_coef))

        ######################################
        self.n_all_actions = self.n_actions + self.n_concepts

    def step(self, env_input):
        env_action = None
        probs = None
        env_input = model_utils.convert_env_input(env_input)
        self.inputs.append(env_input.squeeze(0))
        while env_action is None:

            probs, estimates, concept_probs, encoder_esimate = self.ac.forward(env_input, probs)

            action_probs = probs.squeeze(0)
            self.concept_probs.append(concept_probs.squeeze(0))
            # action = np.random.choice(self.n_all_actions, p=action_probs.squeeze(0).detach().cpu().numpy())
            action = np.random.choice(self.n_actions, p=action_probs.squeeze(0).detach().cpu().numpy())

            if action < self.n_actions:
                self.encoder_estimates.append(encoder_esimate.squeeze(0))
                env_action = action
            else:
                self.internal_estimates.append(estimates.squeeze(0) - self.imagination_decay)
                internal_action_taken = action_probs[action]
                self.imagine_taken_probs.append(internal_action_taken)

        # self.ac.prune()
        self.action_probs.append(action_probs)
        self.value_estimates.append(estimates.squeeze(0))
        self.action_taken_probs.append(action_probs[env_action])
        self.t += 1
        return env_action

    def update_policy(self, env_reward, episode_end=True, new_state =None, learn_policy = True, learn_encoding = False):
        ret_loss = {}
        self.rewards.append(env_reward)
        latest_reward = env_reward  # + self.aux_rewards[-1]
        should_update = self.should_update(episode_end, latest_reward)
        if should_update:
            loss = 0
            if learn_policy:
                V_target = [0]
                # rewards = torch.tensor(self.rewards).to(settings.DEVICE)
                # aux_rewards = torch.tensor(self.aux_rewards).to(settings.DEVICE)
                while self.rewards:
                    # latest reward + (future reward * gamma)
                    reward = self.rewards.pop(-1)  # + self.aux_rewards.pop(-1)
                    V_target.insert(0, reward + (self.discount_factor * V_target[0]))
                    # discounted_rewards.insert(0, reward)
                V_target.pop(-1)  # remove the extra 0 placed before the loop

                V_target = torch.tensor(V_target).to(settings.DEVICE)
                V_estimate = torch.cat(self.value_estimates, dim=0)

                # print(torch.sign(advantage))
                if V_target.shape[0] > 1:
                    V_target = (V_target - V_target.mean()) / (V_target.std() + 1e-9)  # normalizing return
                advantage = V_target - V_estimate

                action_prob_vector = torch.stack(self.action_probs)
                taken_action_probs_vector = torch.stack(self.action_taken_probs)

                ## NON AC STUFF

                action_log_prob = torch.log(taken_action_probs_vector)
                actor_loss = (-action_log_prob * advantage.detach()).sum()
                entropy_loss = (torch.log(action_prob_vector) * action_prob_vector).mean()

                critic_loss = F.smooth_l1_loss(input=V_estimate, target=V_target,
                                               reduction='mean')

                # concept_prob_vector = torch.stack(self.concept_probs)
                # concept_entropy_loss = (torch.log(concept_prob_vector) * concept_prob_vector).mean()

                ret_loss['actor_loss'] = actor_loss.detach().cpu().numpy()
                ret_loss['entropy_loss'] = entropy_loss.detach().cpu().numpy()
                ret_loss['critic_loss'] = critic_loss.detach().cpu().numpy()
                # ret_loss['concept_entropy_loss'] = concept_entropy_loss.detach().cpu().numpy()

                # loss += (self.entropy_coef * concept_entropy_loss)
                loss += actor_loss + critic_loss + (self.entropy_coef * entropy_loss)
                # print(actor_loss, ' ', critic_loss)

            if learn_encoding:
                inputs_vector = torch.stack(self.inputs)
                encoder_esimates = torch.stack(self.encoder_estimates)

                encoder_loss = F.smooth_l1_loss(input=encoder_esimates, target=inputs_vector,
                                                reduction='mean')
                ret_loss['encoder_loss'] = encoder_loss.detach().cpu().numpy()
                # loss += encoder_loss

            if len(self.internal_estimates) > 0:
                print(exit(9))
                print(len(self.internal_estimates))
                internal_estimates = torch.cat(self.internal_estimates, dim=0)
                reward_tensor = torch.tensor([env_reward]).to(settings.DEVICE).repeat(1, len(self.internal_estimates))

                V_estimate = .1 * internal_estimates + .9 * reward_tensor

                imagine_taken_probs_vector = torch.stack(self.imagine_taken_probs)
                action_log_prob = torch.log(imagine_taken_probs_vector)
                internal_loss = (-action_log_prob * V_estimate).mean()  # + self.learnable_variance[0]
                internal_loss.backward(retain_graph=True)

            loss.backward(retain_graph=True)
            self.update_networks()
            self.reset_buffers()
        return ret_loss

    def update_networks(self):
        self.ac.update_parameters()
        # self.ac.reset_state()

    def should_update(self, episode_end, reward):
        steps_since_update = len(self.rewards) + 1
        td_update = self.td_step != -1 and steps_since_update % self.td_step == 0
        if self.update_threshold == -1:  # not trying the threshold updater
            return episode_end or td_update
        update = episode_end or np.abs(reward) >= self.update_threshold
        return update

    def reset_buffers(self):
        self.inputs = []
        self.rewards = []
        self.value_estimates = []
        self.action_probs = []
        self.action_taken_probs = []

        self.concept_probs = []
        self.encoder_estimates = []
        self.imagine_probs = []
        self.imagine_taken_probs = []
        self.internal_estimates = []

    def get_focus(self):
        return self.pred_val  # need to make a difference if to self or to environment

    def get_reward_perception(self):
        return self.pred_feel_val

    def get_env_pred_val(self):
        return self.pred_val if self.pred < self.out_vector_idx[len(self.action_channels)] else None

    def get_action(self):
        return self.get_env_pred_val()  # todo we want to implement this in the agent i think...

    def log_predictions(self, writer=sys.stdout):
        writer.write('\nAgent Summary at timestep ' + str(self.t) + '\n')
        writer.write('prediction to environment: ' + str(self.model.get_env_pred_val()) + '\n')
        writer.write(str(self.pred_val) + ', ' + str(self.pred_feel_val))
        writer.write('\n\n full reward is: ' + str(self.reward))
        writer.write('\n')
        writer.flush()

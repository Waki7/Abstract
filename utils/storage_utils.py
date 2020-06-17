import logging
import os
import re
from datetime import datetime
from typing import Iterable, Union

import gym
import numpy as np
import yaml
from tensorboardX import SummaryWriter

import settings
import utils.env_wrappers as env_wrappers


def clean_experiment_folders():
    experiment_folder = settings.LOG_DIR
    for subdir, dirs, files in os.walk(experiment_folder):
        for dir in dirs:
            if re.match('\d\d\d\d-\d\d', dir):
                print(dir)


class ExperimentLogger():
    def __init__(self):
        self.results_path = ''
        self.writer = None
        self.progress_values_sum = {}
        self.progress_values_mean = {}
        self.counts = {}

    def reset_buffers(self, reset_count):
        # reset progress buffer after every progress update
        self.progress_values_mean = {}
        self.progress_values_sum = {}
        if reset_count:
            self.counts = {}

    def create_experiment(self, agent_name, env_name, training_cfg, directory='', agent_cfg=None, env_cfg=None):
        variation = training_cfg.get('variation', '')
        training = directory == ''
        if not training:
            # todo add a testing directory to this
            self.results_path = directory
        else:
            time = datetime.now()
            self.results_path = '{}/{}/{}/{}{}'.format(settings.LOG_DIR, agent_name, env_name, variation,
                                                       time.strftime("%Y_%m%d_%H%M"))
            logging.info(self.results_path)
            self.writer = SummaryWriter(self.results_path)
            # env.unwrapped.spec.id

        # ----------------------------------------------------------------
        # create empty notes file
        # ----------------------------------------------------------------
        notes_file = '{}/notes.txt'.format(self.results_path)
        open(notes_file, 'a').close()

        # ----------------------------------------------------------------
        # store any passed in configs
        # ----------------------------------------------------------------
        if agent_cfg is not None:
            with open('{}/agent.yaml'.format(self.results_path), 'w') as file:
                yaml.dump(agent_cfg, file)
        if env_cfg is not None:
            with open('{}/env.yaml'.format(self.results_path), 'w') as file:
                yaml.dump(env_cfg, file)

        self.reset_buffers(True)

    def create_sub_experiment(self):
        pass

    def log_progress(self, episode, step):
        log_output = "episode: {}, step: {}  ".format(episode, step)
        for key in self.progress_values_mean.keys():
            label = 'average_{}'.format(key)
            mean = np.round(np.mean(self.progress_values_mean[key]), decimals=3)
            log_output += '{}: {} , '.format(key, mean)
            self.writer.add_scalar(label, mean, global_step=episode)

        for key in self.progress_values_sum.keys():
            label = 'total_{}'.format(key)
            sum = np.round(np.sum(self.progress_values_sum[key]), decimals=3)
            log_output += '{}: {} , '.format(key, sum)
            self.writer.add_scalar(label, sum, global_step=episode)

        logging.info(log_output)
        self.reset_buffers(False)

    def add_scalar_dict(self, label, data, step=-1, track_mean=False, track_sum=False, log=False):
        if len(data) == 0:
            return
        keys = list(data.keys())
        if 'agent' in keys[0]:
            # average over the agents
            raise NotImplementedError('have the controller combine agent losses')
            pass
        for key in keys:
            self.add_agent_scalars(label='{}/{}'.format(label, key), data=data[key], step=step,
                                   track_mean=track_mean, track_sum=track_sum,
                                   log=log)

    def add_agent_scalars(self, label, data, step=-1, track_mean=False, track_sum=False, log=False):
        if data is None:
            return
        if isinstance(data, Iterable):
            data = np.mean(data)

        if track_mean:
            self.progress_values_mean[label] = self.progress_values_mean.get(label, [])
            self.progress_values_mean[label].append(data)
        if track_sum:
            self.progress_values_sum[label] = self.progress_values_sum.get(label, [])
            self.progress_values_sum[label].append(data)

        if log:
            if step == -1:
                self.counts[label] = self.counts.get(label, 0)
                self.counts[label] += 1
                step = self.counts[label]
            self.writer.add_scalar(label, data, global_step=step)

    def checkpoint(self, episode, checkpoint_freq, environment: Union[env_wrappers.SubprocVecEnv, gym.Env]):
        if (episode + 1) % checkpoint_freq == 0:
            # --- save models
            pass

            # --- save animations
            animations = environment.render()
            animations_folder = '{}/environment_episode_{}'.format(self.results_path, episode)

            return None

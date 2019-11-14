import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter
import settings
import logging
import os
import re


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

    def create_experiment(self, agent_name, env_name, training_cfg, directory=''):
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
        self.reset_buffers(True)

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

    def add_agent_scalars(self, label, data, step=-1, track_mean=False, track_sum=False, log=False):
        if data is None:
            return
        if isinstance(data, list):
            data = np.mean(data)
        if isinstance(data, dict):
            data = np.mean(list(data.values()))
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

    def checkpoint(self):
        pass

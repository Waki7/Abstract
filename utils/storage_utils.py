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
        self.progress_values = {}

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

    def log_progress(self, episode, step):
        log_output = "episode: {}, step: {}".format(episode, step)

        for key in self.progress_values.keys():
            log_output += 'average_{}: {}, total_{}: {}, '.format(key, np.round(np.mean(self.progress_values[key]),
                                                                                decimals=3),
                                                                  key, np.round(np.sum(self.progress_values[key]),
                                                                                decimals=3))
        logging.info(log_output)
        self.progress_values = {}  # reset progress buffer after every progress update

    def add_agent_scalars(self, label, val, track_locally=False):
        if val is None:
            return
        if isinstance(val, list):
            val = np.mean(val)
        if isinstance(val, dict):
            val = np.mean(list(val.values()))
        if track_locally:
            self.progress_values[label] = self.progress_values.get(label, [])
            self.progress_values[label].append(val)
        self.writer.add_scalar(label, val)

    def checkpoint(self):
        pass

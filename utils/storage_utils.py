import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter
import settings
import logging


class ExperimentLogger():
    def __init__(self):
        self.result_path = ''
        self.writer = None
        self.progress_values = {}

    def create_experiment(self, agent_name, env_name, directory=''):
        if directory == '':
            # todo add a testing directory to this
            self.result_path = directory

        else:
            self.writer = SummaryWriter(self.result_path)
            time = datetime.now()
            env_id = (
                self.env.unwrapped.spec.id if not hasattr(self.env, 'envs') else self.env.envs[0].unwrapped.spec.id)
            results_path = '{}/{}/{}/{}'.format(settings.LOG_DIR, agent_name, env_name, time.strftime("%Y%%m%d_%H%M"))


    def log_progress(self, episode, step):
        log_output = '\n'
        log_output += "episode: {}, step: {}".format(episode, step)

        for key in self.progress_values.keys():
            log_output += 'average_{}: {}, total_{}: {}, '.format(key, np.round(np.mean(self.progress_values[key]),
                                                                                decimals=3),
                                                                  np.round(np.sum(self.progress_values[key]),
                                                                           decimals=3))

    def add_agent_scalars(self, label, val, track_locally=False):
        val = val
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

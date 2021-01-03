import inspect
import logging
import os
import re
from datetime import datetime
from typing import *

import gym
import numpy as np
from array2gif import write_gif
from tensorboardX import SummaryWriter

import agent_algorithms as agnts
import networks as nets
import utils.env_wrappers as env_wrappers
import utils.storage_utils as storage_utils
from utils.paths import Directories


def clean_experiment_folders():
    experiment_folder = Directories.LOG_DIR
    for subdir, dirs, files in os.walk(experiment_folder):
        for dir in dirs:
            if re.match('\d\d\d\d-\d\d', dir):
                print(dir)


def copy_config_param(src_cfg, target_cfg, param_name, fallback_value=None):
    '''
    will update the target cfg with the param specified from the src cfg, target cfg will be updated by reference,
    will only return the original param value
    :param src_cfg:
    :param target_cfg:
    :param param_name:
    :param fallback_value:
    :return:
    '''
    val = src_cfg.get(param_name, fallback_value)
    logging.debug(' {} : {}'.format(param_name, val))
    target_cfg[param_name] = val
    return val


def get_experiment_naming(algo):
    algo_name = algo.__class__.__name__
    if isinstance(algo, agnts.Agent):
        root_dir = agnts.Agent.__name__
    elif isinstance(algo, nets.NetworkInterface):
        root_dir = 'EncodingTasks'
    else:
        root_dir = inspect.getmro(algo)[-2]
        logging.info(
            "didn't find expected base class, will store log results directory for {},"
            " the top parent class after object".format(root_dir))
    return root_dir, algo_name


class ExperimentLogger(object):
    def __init__(self, dir_to_continue: str = None):
        self.continuation = dir_to_continue is not None
        self.experiment_root: Optional[
            str] = None if dir_to_continue is None else dir_to_continue
        self.run_dir: Optional[str] = None

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

    @property
    def run_path(self):
        return os.path.join(self.experiment_root, self.run_dir)

    def create_experiment(self, algo: object, run_dir,
                          config_dict: Dict[str, Dict] = {}):
        '''

        :param algo: either pass in a string for the variation of algorithm to make experiment under
                    or pass in an object and we will either map a naming if we have it, or
                    use the class name
        :param run_dir: just the iteration of experimentation to record, so for first training use
                        something like 'training'
        :param config_dict: dictionary of configs to store in the folder
        :return:
        '''
        if self.experiment_root is None:
            root_dir, algo_name = get_experiment_naming(algo)
            time = datetime.now()
            self.experiment_root = os.path.join(Directories.LOG_DIR, root_dir,
                                                algo_name,
                                                time.strftime(
                                                    "%Y_%m%d_%H%M_%S"))
            self.run_dir = run_dir
        logging.info('experiment being logged in {}'.format(self.run_path))
        self.writer = SummaryWriter(self.run_path)

        # ----------------------------------------------------------------
        # create empty notes file, directory for weights, models, and animations
        # ----------------------------------------------------------------
        notes_file = '{}/notes.txt'.format(self.run_path)
        open(notes_file, 'a').close()
        [os.mkdir('{}/{}'.format(self.run_path, folder)) for folder in
         ['weights', 'models', 'animations']]

        # ----------------------------------------------------------------
        # store any passed in configs
        # ----------------------------------------------------------------
        for name, config, in config_dict.items():
            storage_utils.save_config(cfg=config, dir=self.run_path,
                                      filename=name)

        self.reset_buffers(True)

    def create_sub_experiment(self):
        # todo for different tests and testing results
        pass

    def log_progress(self, episode, step):
        log_output = "episode: {}, step: {}  ".format(episode, step)
        for key in self.progress_values_mean.keys():
            label = 'average_{}'.format(key)
            mean = np.round(np.mean(self.progress_values_mean[key]), decimals=3)
            log_output += '{}: {} , '.format(label, mean)
            self.writer.add_scalar(label, mean, global_step=episode)

        for key in self.progress_values_sum.keys():
            label = 'total_{}'.format(key)
            sum = np.round(np.sum(self.progress_values_sum[key]), decimals=3)
            log_output += '{}: {} , '.format(label, sum)
            self.writer.add_scalar(label, sum, global_step=episode)

        logging.info(log_output)
        self.reset_buffers(False)

    def add_scalar_dict(self, label, data, step=-1, track_mean=False,
                        track_sum=False, log=False):
        if len(data) == 0:
            return
        keys = list(data.keys())
        if 'agent' in keys[0]:
            # average over the agents
            raise NotImplementedError(
                'have the controller combine agent losses')
            pass
        for key in keys:
            self.add_agent_scalars(label='{}/{}'.format(label, key),
                                   data=data[key], step=step,
                                   track_mean=track_mean, track_sum=track_sum,
                                   log=log)

    def add_agent_scalars(self, label: str, data: Union[float, Iterable[float]],
                          step: int = -1, track_mean: bool = False,
                          track_sum: bool = False,
                          log: bool = False):
        if data is None:
            return
        if isinstance(data, Iterable):
            data = np.mean(data)

        if track_mean:
            self.progress_values_mean[label] = self.progress_values_mean.get(
                label, [])
            self.progress_values_mean[label].append(data)
        if track_sum:
            self.progress_values_sum[label] = self.progress_values_sum.get(
                label, [])
            self.progress_values_sum[label].append(data)

        if log:
            if step == -1:
                self.counts[label] = self.counts.get(label, 0)
                self.counts[label] += 1
                step = self.counts[label]
            self.writer.add_scalar(label, data, global_step=step)

    # def custom_checkpoint(self, checkpoint_freq, func: tp.Callable[[str, ], None]):
    #     pass
    #     func(self.)

    def checkpoint(self, episode, checkpoint_freq, agent_map,
                   environment: Union[env_wrappers.SubprocVecEnv, gym.Env]):
        is_batch_env = isinstance(environment, env_wrappers.SubprocVecEnv)
        if (episode + 1) % checkpoint_freq == 0:
            # --- save models
            pass

            # --- save animations
            env_animations_path = '{}/animations/environment_episode_{}.gif'.format(
                self.run_path, episode)

            if is_batch_env:
                animations = environment.render(indices=(0,))[0]
            else:
                animations = environment.render()
            write_gif(animations, env_animations_path, fps=2)

            if hasattr(environment, 'render_agent_pov') or (
                    is_batch_env and environment.has_attr('render_agent_pov')):
                for agent_key in agent_map.keys():
                    agent_animation_path = '{}/animations/agent_{}_episode_{}.gif'.format(
                        self.run_path,
                        agent_key, episode)

                    if is_batch_env:
                        agent_animation = \
                            environment.env_method('render_agent_pov',
                                                   *(agent_key,), indices=0)[0]
                    else:
                        agent_animation = environment.render_agent_pov(
                            agent_key)
                    write_gif(agent_animation, agent_animation_path, fps=2)

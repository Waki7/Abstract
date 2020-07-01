"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import random
from multiprocessing import Process, Pipe

import numpy as np
import torch
from baselines.common.vec_env import VecEnv, CloudpickleWrapper

import settings


def worker(remote, parent_remote, env_fn_wrapper, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'render':
            rgb_frames = env.render()
            remote.send(rgb_frames)
        elif cmd == 'env_method':
            method = getattr(env, data[0])
            remote.send(method(*data[1], **data[2]))
        elif cmd == 'get_attr':
            remote.send(getattr(env, data))
        elif cmd == 'set_attr':
            remote.send(setattr(env, data[0], data[1]))
        elif cmd == 'get_unwrapped':
            remote.send(env)
        else:
            raise NotImplementedError("`{}` is not implemented in the worker".format(cmd))


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        seeds = np.arange(nenvs) + settings.SEED
        self.ps = [Process(target=worker,
                           args=(self.work_remotes[i], self.remotes[i], CloudpickleWrapper(env_fns[i]), seeds[i]))
                   for i in range(nenvs)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        observations, rewards, dones, infos = zip(*results)
        return observations, rewards, dones, infos

    def render(self, indices=None):
        '''
        By default we only want to render one enviornment's animations, you can pass None to render all
        :param indices:
        :return:
        '''
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('render', None))
        animations = [remote.recv() for remote in target_remotes]
        return animations

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        a = [remote.recv() for remote in self.remotes]
        return a

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('set_attr', (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('env_method', (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def get_unwrapped(self, indices=None):
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('get_unwrapped', ()))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices):
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: ([multiprocessing.Connection]) Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]

    def _get_indices(self, indices):
        """
        Convert a flexibly-typed reference to environment indices to an implied list of indices.

        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: (list) the implied list of indices.
        """
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        return indices

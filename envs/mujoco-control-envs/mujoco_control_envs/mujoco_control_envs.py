import sys
import os

from .tp_envs.half_cheetah_vel import HalfCheetahVelEnv as HalfCheetahVelEnv_
from .tp_envs.half_cheetah_dir import HalfCheetahDirEnv as HalfCheetahDirEnv_
from .tp_envs.ant_dir import AntDirEnv as AntDirEnv_
from .tp_envs.walker_rand_params_wrapper import WalkerRandParamsWrappedEnv as WalkerRandParamsWrappedEnv_

from collections import namedtuple
import json, pickle, os
import numpy as np
from typing import List
from gym.spaces import Box
from gym.wrappers import TimeLimit

class HalfCheetahDirEnv(HalfCheetahDirEnv_):
    def __init__(self, tasks: List[dict], include_goal: bool = False):
        self.include_goal = include_goal
        super(HalfCheetahDirEnv, self).__init__()
        if tasks is None:
            tasks = [{'direction': 1}, {'direction': -1}]
        self.tasks = tasks
        self.set_task_idx(0)
        self._max_episode_steps = 200

    def _get_obs(self):
        if self.include_goal:
            idx = 0
            try:
                idx = self.tasks.index(self._task)
            except:
                pass
            one_hot = np.zeros(len(self.tasks), dtype=np.float32)
            one_hot[idx] = 1.0
            obs = super()._get_obs()
            obs = np.concatenate([obs, one_hot])
        else:
            obs = super()._get_obs()
        return obs
    
    def set_task(self, task):
        self._task = task
        self._goal_dir = self._task['direction']
        self.reset()

    def set_task_idx(self, idx):
        self.set_task(self.tasks[idx])

class HalfCheetahVelEnv(HalfCheetahVelEnv_):
    def __init__(self, tasks: List[dict] = None, include_goal: bool = False, one_hot_goal: bool = False, n_tasks: int = None):
        self.include_goal = include_goal
        self.one_hot_goal = one_hot_goal
        if tasks is None:
            assert n_tasks is not None, "Either tasks or n_tasks must be non-None"
            tasks = self.sample_tasks(n_tasks)
        self.n_tasks = len(tasks)
        super().__init__(tasks)
        self.set_task_idx(0)
        self._max_episode_steps = 200

    def _get_obs(self):
        if self.include_goal:
            obs = super()._get_obs()
            if self.one_hot_goal:
                goal = np.zeros((self.n_tasks,))
                goal[self.tasks.index(self._task)] = 1
            else:
                goal = np.array([self._goal_vel])
            obs = np.concatenate([obs, goal])
        else:
            obs = super()._get_obs()

        return obs
        
    def set_task(self, task):
        self._task = task
        self._goal_vel = self._task['velocity']
        self.reset()

    def set_task_idx(self, idx):
        self.task_idx = idx
        self.set_task(self.tasks[idx])

class AntDirEnv(AntDirEnv_):
    def __init__(self, tasks: List[dict], n_tasks: int = None, include_goal: bool = False):
        self.include_goal = include_goal
        super(AntDirEnv, self).__init__(forward_backward=n_tasks == 2)
        if tasks is None:
            assert n_tasks is not None, "Either tasks or n_tasks must be non-None"
            tasks = self.sample_tasks(n_tasks)
        self.tasks = tasks
        self.n_tasks = len(self.tasks)
        self.set_task_idx(0)
        self._max_episode_steps = 200
    
    def _get_obs(self):
        if self.include_goal:
            idx = 0
            try:
                idx = self.tasks.index(self._task)
            except:
                pass
            one_hot = np.zeros(50, dtype=np.float32)
            one_hot[idx] = 1.0
            obs = super()._get_obs()
            obs = np.concatenate([obs, one_hot])
        else:
            obs = super()._get_obs()
        return obs
    
    def set_task(self, task):
        self._task = task
        self._goal = task['goal']
        self.reset()

    def set_task_idx(self, idx):
        self.set_task(self.tasks[idx])

class WalkerRandParamsWrappedEnv(WalkerRandParamsWrappedEnv_):
    def __init__(self, tasks: List[dict] = None, n_tasks: int = None, include_goal: bool = False):
        self.include_goal = include_goal
        self.n_tasks = len(tasks) if tasks is not None else n_tasks
        
        # sample tasks here
        super(WalkerRandParamsWrappedEnv, self).__init__(tasks, n_tasks)

        self.set_task_idx(0)
        self._max_episode_steps = 200
        
    def _get_obs(self):
        if self.include_goal:
            idx = 0
            try:
                idx = self._goal
            except:
                pass
            one_hot = np.zeros(self.n_tasks, dtype=np.float32)
            one_hot[idx] = 1.0
            obs = super()._get_obs()
            obs = np.concatenate([obs, one_hot])
        else:
            obs = super()._get_obs()
        return obs
        
    # set to a new task by its id and reset the env
    def set_task_idx(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()


class ML45Env(object):
    def __init__(self, include_goal: bool = False):
        self.n_tasks = 50
        self.tasks = list(HARD_MODE_ARGS_KWARGS['train'].keys()) + list(HARD_MODE_ARGS_KWARGS['test'].keys())

        self._max_episode_steps = 150

        self.include_goal = include_goal
        self._task_idx = None
        self._env = None
        self._envs = []

        _cls_dict = {**HARD_MODE_CLS_DICT['train'], **HARD_MODE_CLS_DICT['test']}
        _args_kwargs = {**HARD_MODE_ARGS_KWARGS['train'], **HARD_MODE_ARGS_KWARGS['test']}
        for idx in range(self.n_tasks):
            task = self.tasks[idx]
            args_kwargs = _args_kwargs[task]
            if idx == 28 or idx == 29:
                args_kwargs['kwargs']['obs_type'] = 'plain'
                args_kwargs['kwargs']['random_init'] = False
            else:
                args_kwargs['kwargs']['obs_type'] = 'with_goal'
            args_kwargs['task'] = task
            env = _cls_dict[task](*args_kwargs['args'], **args_kwargs['kwargs'])
            self._envs.append(TimeLimit(env, max_episode_steps=self._max_episode_steps))
        
        self.set_task_idx(0)

    @property
    def observation_space(self):
        space = self._env.observation_space
        if self.include_goal:
            space = Box(low=space.low[0], high=space.high[0], shape=(space.shape[0] + len(self.tasks),))
        return space

    def reset(self):
        obs = self._env.reset()
        if self.include_goal:
            one_hot = np.zeros(len(self.tasks), dtype=np.float32)
            one_hot[self._task_idx] = 1.0
            obs = np.concatenate([obs, one_hot])
        return obs

    def step(self, action):
        o, r, d, i = self._env.step(action)
        if self.include_goal:
            one_hot = np.zeros(len(self.tasks), dtype=np.float32)
            one_hot[self._task_idx] = 1.0
            o = np.concatenate([o, one_hot])
        return o, r, d, i

    def set_task_idx(self, idx):
        self._task_idx = idx
        self._env = self._envs[idx]

    def __getattribute__(self, name):
        '''
        If we try to access attributes that only exist in the env, return the
        env implementation.
        '''
        try:
            return object.__getattribute__(self, name)
        except AttributeError as e:
            e_ = e
            try:
                return object.__getattribute__(self._env, name)
            except AttributeError as env_exception:
                pass
            except Exception as env_exception:
                e_ = env_exception
        raise e_
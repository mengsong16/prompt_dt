import numpy as np
import gym
import json, pickle, random, os, torch
from collections import namedtuple
from prompt_dt.prompt_utils import load_train_test_env_name_list
from prompt_dt.utils.path import *
from prompt_dt.prompt_utils import get_env_goal

# for mujoco tasks
from mujoco_control_envs.mujoco_control_envs import HalfCheetahDirEnv, HalfCheetahVelEnv, AntDirEnv, WalkerRandParamsWrappedEnv
# for metaworld
import metaworld
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)


# create a single environment with reward scale, maximum episode length, return target
def create_env(env_name, config_save_path, seed=1):
    if 'cheetah_dir' in env_name:
        # include_goal = False: do not include goal in observations
        if '0' in env_name:  # direction 1
            # tasks = [{'direction': 1}]
            env = HalfCheetahDirEnv([{'direction': 1}], include_goal = False)
        elif '1' in env_name: # direction -1
            env = HalfCheetahDirEnv([{'direction': -1}], include_goal = False)
        
        #print(env._goal)  # 1
        
    elif 'cheetah_vel' in env_name:
        task_idx = int(env_name.split('-')[-1])
        task_paths = f"{config_save_path}/cheetah_vel/config_cheetah_vel_task{task_idx}.pkl"
        tasks = []
        with open(task_paths.format(task_idx), 'rb') as f:
            task_info = pickle.load(f)
            assert len(task_info) == 1, f'Unexpected task info: {task_info}'
            tasks.append(task_info[0])
        # print(tasks[0])  # env._goal = 0.075
        # include_goal = False: do not include goal in observations
        env = HalfCheetahVelEnv(tasks, include_goal = False)
        
    elif 'ant_dir' in env_name:
        task_idx = int(env_name.split('-')[-1])
        task_paths = f"{config_save_path}/ant_dir/config_ant_dir_task{task_idx}.pkl"
        tasks = []
        with open(task_paths.format(task_idx), 'rb') as f:
            task_info = pickle.load(f)
            assert len(task_info) == 1, f'Unexpected task info: {task_info}'
            tasks.append(task_info[0])
        # print(tasks[0]) # {'goal': 1.2033605945679715} 
        # env._goal = 1.2033605945679715
        # include_goal = False: do not include goal in observations
        env = AntDirEnv(tasks, len(tasks), include_goal = False)
        
    elif 'ML1-' in env_name: # metaworld ML1
        task_name = '-'.join(env_name.split('-')[1:-1])
        ml1 = metaworld.ML1(task_name, seed=seed) # construct the benchmark, sampling tasks
        env = ml1.train_classes[task_name]()  # create an environment with task
        task_idx = int(env_name.split('-')[-1])
        task = ml1.train_tasks[task_idx]
        env.set_task(task)  # set task
        # print(env.goal) [0.1 0.8 0.2]
        
    else:
        raise NotImplementedError
    
    return env

def create_env_list(env_name_list, config_save_path):
    env_list = []
    
    for env_name in env_name_list:
        env = create_env(env_name=env_name, config_save_path=config_save_path)
        env_list.append(env)
    return env_list

def test_envs(render):
    #base_env = 'cheetah_vel' # ['cheetah_dir', 'cheetah_vel', 'ant_dir', 'ML1-pick-place-v2']
    #train_env_name_list, test_env_name_list = load_train_test_env_name_list(base_env)

    # 'ant_dir-0', 'cheetah_vel-0', 'cheetah_dir-0', 'ML1-pick-place-v2-0'
    env_name = 'cheetah_vel-0'
    env = create_env(env_name=env_name, config_save_path=task_config_path)
    if "ML1" in env_name:
        max_ep_len = 500
    else:
        max_ep_len = 200

    # suppress scientific notation
    np.set_printoptions(suppress=True)
    # set print precision
    np.set_printoptions(precision=2)


    for episode in range(100):
        print("--------------------------------------")
        print('Episode: {}'.format(episode))
        obs = env.reset()
        print("state s0: %s"%obs)

        for i in range(max_ep_len): 
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            #print('action: {}   state: {}  reward: {}'.format(action, obs, reward))
            if i == 0:
                print("state s1: %s"%obs)

            if render:
                env.render()

            if done: 
                print("last state: %s"%(obs))
                break
                    
        print('Episode {} finished after {} timesteps.'.format(episode,i+1))
    
    print('-----------------------------')
    print("Observation space: ", env.observation_space)
    print("Action space: ", env.action_space)
    print("Goal: ", get_env_goal(env_name, env))
    print('-----------------------------')

def test_walker():
    env = WalkerRandParamsWrappedEnv(n_tasks=50)
    # print goal parameters
    print(env.tasks[0])
    
    for task_id in range(50):
        env.reset()
        env.set_task_idx(task_id)
        
        # test in current env for 100 steps
        for step in range(100):
            env.render()
            # take a random action
            env.step(env.action_space.sample())
            print(step)  


if __name__ == '__main__':
    #test_envs(render=False)
    test_walker()


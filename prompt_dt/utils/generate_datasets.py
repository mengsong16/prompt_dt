import sys
from pathlib import Path
import gym
import numpy as np
import os
import pickle
import torch
import json
from collections import namedtuple
from prompt_dt.prompt_utils import load_train_test_env_name_list, load_data_prompt
from prompt_dt.utils.other import seed_other
from prompt_dt.utils.path import *

# split: ["train", "test"]
def load_env_name_list(base_env_name, split):
    task_config = os.path.join(task_config_path, config_path_dict[base_env_name])
    with open(task_config, 'r') as f:
        task_config = json.load(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    
    env_name_list = []
    if split == "train":
        task_list = task_config.train_tasks
    else:
        task_list = task_config.test_tasks
    
    for task_ind in task_list:
        env_name_list.append(base_env_name +'-'+ str(task_ind))
    

    print("======================== %s %s envs:%d =============================="%(base_env_name, split, len(env_name_list)))
    print(env_name_list)
   
    return env_name_list

def gen_goal_state_datasets(seed=1):
    # seed everything except environments
    seed_other(seed)
    splits = ["train", "test"]
    base_envs = ['cheetah_dir', 'cheetah_vel', 'ant_dir', 'ML1-pick-place-v2']
    traj_types = ["input", "prompt"]
    for base_env in base_envs:
        for split in splits:
            # loat env names in train/test split
            env_name_list = load_env_name_list(base_env, split)
            for traj_type in traj_types:
                gen_one_goal_state_dataset(base_env, split, env_name_list, traj_type)
    

def load_trajectories(env_name_list, data_path, dataset_mode, base_env, traj_type):
    trajectory_list = [] 
    for env_name in env_name_list:
        if traj_type == "input":
            dataset_path = data_path + f'/{base_env}/{env_name}-{dataset_mode}.pkl'
        else:
            dataset_path = data_path + f'/{base_env}/{env_name}-prompt-{dataset_mode}.pkl'
        
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
            trajectory_list.extend(trajectories)
    
    return trajectory_list
    
# dataset_mode: ["expert"] 
# traj_type: ["input", "prompt"]
# Note that other modes are not provided in the data)
def gen_one_goal_state_dataset(base_env, split, env_name_list, traj_type,
                               dataset_mode="expert"):
    
    # load trajectories
    trajectory_list = load_trajectories(env_name_list, data_path, dataset_mode, base_env, traj_type)
    print("======> Loaded %d trajectories from %s %s %s %s"%(len(trajectory_list), base_env, dataset_mode, split, traj_type))
    
    goal_states = []
    for traj in trajectory_list:
        # (state_dim, )
        goal_state = traj['observations'][-1]
        goal_states.append(goal_state)
    
    # save goal states
    goal_state_folder = os.path.join(data_path, base_env, "goal_states")
    if not os.path.exists(goal_state_folder):
            os.makedirs(goal_state_folder) 
    
    if traj_type == "input":
        traj_type_str = ""
    else:
        traj_type_str = "-prompt"
    
    goal_state_file_name = f'{base_env}{traj_type_str}-{dataset_mode}-{split}-goal_states.pkl'
    save_path = os.path.join(goal_state_folder, goal_state_file_name)
    
    with open(save_path, 'wb') as f:
        pickle.dump(goal_states, f)
    
    print('======> Goal states saved to ', save_path)



if __name__ == "__main__": 
    # ['cheetah_dir', 'cheetah_vel', 'ant_dir', 'ML1-pick-place-v2']
    gen_goal_state_datasets()

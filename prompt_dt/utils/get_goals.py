import gym
import numpy as np
import torch
import wandb
from prompt_dt.prompt_utils import get_env_goal, gen_env
from prompt_dt.utils.other import parse_config, seed_other
from prompt_dt.utils.gen_goal_state_datasets import load_env_name_list
from prompt_dt.utils.path import *

def get_goals(base_env, split, seed):

    # loat env names in train/test split
    env_name_list = load_env_name_list(base_env, split)
    for env_name in env_name_list:
        # create env
        env, _, _, _ = gen_env(env_name=env_name, config_save_path=task_config_path, seed=seed)
        # goal: (goal_dim, )
        env_goal = get_env_goal(env_name, env)
        
        print("-"*80)
        print("%s: %s"%(env_name, env_goal))

def get_goals_all(seed=1):
    # seed everything except environments
    seed_other(seed)
    splits = ["train", "test"]
    base_envs = ['cheetah_dir', 'cheetah_vel', 'ant_dir', 'ML1-pick-place-v2']
    for base_env in base_envs:
        for split in splits:
            get_goals(base_env, split, seed)


if __name__ == "__main__":
    get_goals_all() 



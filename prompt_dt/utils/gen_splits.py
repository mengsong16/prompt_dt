import numpy as np
import gym
import json, pickle, random, os, torch
from collections import namedtuple
from prompt_dt.utils.path import *
from prompt_dt.prompt_utils import load_train_test_env_name_list
from prompt_dt.utils.other import seed_other
import random
import metaworld

def gen_train_test_split_single_task(base_env, seed, total_task_num: int, test_task_num: int, save_path=None):
    # seed everything except environments
    seed_other(seed=seed)

    task_config = {"env": base_env, "total_tasks": total_task_num}

    # split single task
    total_tasks_indices = list(range(total_task_num))
    # uniformly sample without replacement
    test_task_indices = random.sample(total_tasks_indices, k=test_task_num)
    train_task_indices = list(set(total_tasks_indices) - set(test_task_indices))
    task_config["train_tasks"] = train_task_indices
    task_config["test_tasks"] = test_task_indices
    
    # for single task, refer path to task_config_path
    # for multi task, refer to given path
    # assume the task_config folder already exists
    if save_path is None:
        save_task_config_path = os.path.join(task_config_path, config_path_dict[base_env])
    else:
        save_task_config_path = os.path.join(task_config_path, save_path)

    with open(save_task_config_path, 'w') as f:
        json.dump(task_config, f)
    
    print("======> Train/test split saved to ", save_task_config_path)
    print("======> Done with train/test splits")
    print(task_config)

    
    
# for ML10
def gen_train_test_split_multitask(base_env, total_task_num: int, test_task_num: int):
    ml10 = metaworld.ML10(seed=1)
    env_names = [] 
    for env_name, env_cls in ml10.train_classes.items():
        env_names.append(f'{base_env}-{env_name}')


    for i, env_name in enumerate(env_names):
       gen_train_test_split_single_task(base_env=env_name, seed=i+100, 
                                   total_task_num=total_task_num, 
                                   test_task_num=test_task_num,
                                   save_path=os.path.join(env_name, f'{env_name}-{total_task_num}.json'))


if __name__ == '__main__':
    # use different seed for different base env

    #gen_train_test_split_single_task(base_env="ML1-reach-v2", seed=1, total_task_num=50, test_task_num=5)
    #train_env_name_list, test_env_name_list = load_train_test_env_name_list(env_name="ML1-reach-v2") # verify

    #gen_train_test_split_single_task(base_env="ML1-sweep-v2", seed=2, total_task_num=50, test_task_num=5)
    #train_env_name_list, test_env_name_list = load_train_test_env_name_list(env_name="ML1-sweep-v2") # verify

    #gen_train_test_split_multitask(base_env="ML10", total_task_num=50, test_task_num=5)
    train_env_name_list, test_env_name_list = load_train_test_env_name_list(env_name="ML10") # verify

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
    #test_task_indices = random.sample(total_tasks_indices, k=test_task_num)
    interval = int(total_task_num // test_task_num)
    test_task_indices = []
    start = 0
    for i in range(test_task_num):
        sample_range = list(range(start, start+interval))
        cur_sample = np.random.choice(sample_range, 1, replace=False)[0]
        # from int64 to int32
        cur_sample = int(cur_sample)
        test_task_indices.append(cur_sample)
        start += interval
    
    train_task_indices = list(set(total_tasks_indices) - set(test_task_indices))
    task_config["train_tasks"] = train_task_indices
    task_config["test_tasks"] = test_task_indices
    
    # for single task, refer path to task_config_path
    # for multi task, refer to given path
    # assume the task_config folder already exists
    if save_path is None:
        if "ML1" in base_env or "ML10" in base_env:
            save_task_config_path = os.path.join(task_config_path, base_env, f'{base_env}-50.json')
        else:
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
    # Must use different seeds for different base envs

    # gen_train_test_split_single_task(base_env="ML1-pick-place-v2", seed=1, total_task_num=50, test_task_num=5)
    # train_env_name_list, test_env_name_list = load_train_test_env_name_list(env_name="ML1-pick-place-v2") # verify

    # gen_train_test_split_single_task(base_env="ML1-push-v2", seed=2, total_task_num=50, test_task_num=5)
    # train_env_name_list, test_env_name_list = load_train_test_env_name_list(env_name="ML1-push-v2") # verify

    # gen_train_test_split_single_task(base_env="walker_param", seed=6, total_task_num=50, test_task_num=5)
    # train_env_name_list, test_env_name_list = load_train_test_env_name_list(env_name="walker_param") # verify

    # gen_train_test_split_multitask(base_env="ML10", total_task_num=50, test_task_num=5)
    # train_env_name_list, test_env_name_list = load_train_test_env_name_list(env_name="ML10") # verify

    gen_train_test_split_single_task(base_env="ML1-reach-v2", seed=8, total_task_num=50, test_task_num=5)
    train_env_name_list, test_env_name_list = load_train_test_env_name_list(env_name="ML1-reach-v2") # verify

    
    
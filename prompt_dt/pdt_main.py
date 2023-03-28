from ast import parse
import gym
import numpy as np
import torch
import wandb

import argparse
import pickle
import random
import sys
import time
import itertools
import datetime
import shutil
import json

from prompt_dt.prompt_decision_transformer import PromptDecisionTransformer
from prompt_dt.prompt_seq_trainer import PromptSequenceTrainer
from prompt_dt.prompt_utils import get_env_list
from prompt_dt.prompt_utils import get_prompt_batch, get_prompt, get_batch, get_goal_prompt_batch, get_goal_prompt
from prompt_dt.prompt_utils import get_total_data_mean_std, load_data_prompt, process_info, load_return_info, replace_target_return, append_return_info
from prompt_dt.prompt_utils import load_train_test_env_name_list, get_total_num_trajectory
from prompt_dt.utils.path import *
from prompt_dt.utils.other import parse_config, seed_other
from prompt_dt.prompt_evaluate_episodes import eval_episodes, save_eval_results

from collections import namedtuple
import json, pickle, os


def experiment_mix_env(config_filename, mode):
    # parse config file
    config_file_path = os.path.join(config_path, config_filename)
    variant = parse_config(config_file_path)
    # print out variant
    print("=========================== variant ==================================")
    print(variant)

    # project name
    project_name = variant['project_name']
    
    # device
    device = variant['device']

    # log
    log_to_wandb = variant['log_to_wandb']

    # base environment name
    base_env = variant['base_env']

    # seed
    seed = variant['seed']
    seed_other(seed) # seed everything except environments


    ######
    # construct train and test environments
    ######
    train_env_name_list, test_env_name_list = load_train_test_env_name_list(base_env)

    # training envs (already written info for each env)
    train_info, train_env_list = get_env_list(train_env_name_list, task_config_path, device, seed)
    
    # test envs (already written info for each env)
    test_info, test_env_list = get_env_list(test_env_name_list, task_config_path, device, seed)

    print("======> Loaded %d train envs"%(len(train_env_list)))
    print("======> Loaded %d test envs"%(len(test_env_list)))
    
    ######
    # load train and test datasets (data + prompt)
    ######
    train_dataset_mode = variant['train_dataset_mode']
    test_dataset_mode = variant['test_dataset_mode']
    train_prompt_mode = variant['train_prompt_mode']
    test_prompt_mode = variant['test_prompt_mode']

    # load training dataset 
    train_trajectories_list, train_prompt_trajectories_list, train_trajectory_num, train_prompt_trajectory_num = load_data_prompt(train_env_name_list, data_path, train_dataset_mode, train_prompt_mode, base_env)
    # load test dataset 
    # test_trajectories are for test time finetune only
    # test_prompt_trajectories are for test time prompt
    test_trajectories_list, test_prompt_trajectories_list, test_trajectory_num, test_prompt_trajectory_num = load_data_prompt(test_env_name_list, data_path, test_dataset_mode, test_prompt_mode, base_env)

    print("======> Loaded train trajectories: %d"%(get_total_num_trajectory(train_trajectory_num)))
    print(train_trajectory_num)
    print("======> Loaded train prompt trajectories: %d"%(get_total_num_trajectory(train_prompt_trajectory_num)))
    print(train_prompt_trajectory_num)
    print("======> Loaded test trajectories: %d"%(get_total_num_trajectory(test_trajectory_num)))
    print(test_trajectory_num)
    print("======> Loaded test prompt trajectories: %d"%(get_total_num_trajectory(test_prompt_trajectory_num)))
    print(test_prompt_trajectory_num)

    ######
    # process train and test datasets
    ######
    # compute state mean and std of all training and test trajectories
    if variant['average_state_mean']:
        train_total = list(itertools.chain.from_iterable(train_trajectories_list))
        test_total = list(itertools.chain.from_iterable(test_trajectories_list))
        total_traj_list = train_total + test_total

        total_state_mean, total_state_std= get_total_data_mean_std(total_traj_list)
        variant['total_state_mean'] = total_state_mean
        variant['total_state_std'] = total_state_std

    reward_mode = variant.get('reward_mode', 'normal')
    pct_traj = float(variant.get('pct_traj', 1.))
    # process train dataset info (continue writing info for each env)
    # print("="*80)
    # print("Train environments info")
    # print("="*80)
    train_info = process_info(train_env_name_list, train_trajectories_list, 
                              train_info, reward_mode, train_dataset_mode, 
                              pct_traj, variant, verbose=False)

    # process test dataset info (continue writing info for each env)
    # print("="*80)
    # print("Test environments info")
    # print("="*80)
    test_info = process_info(test_env_name_list, test_trajectories_list, 
                             test_info, reward_mode, test_dataset_mode, 
                             pct_traj, variant, verbose=False)

    print("======> Train and test trajectories processed ")

    return_info = load_return_info(base_env, verbose=False)
    print("======> Return information loaded")

    # replace the per base env target rtg with per env target rtg
    if variant['max_return_as_target_rtg']:
        replace_target_return(train_info, test_info, train_env_name_list, test_env_name_list, return_info)

    # append max rtg and random rtg to env info
    if variant['compare_normalized_returns']:
        append_return_info(train_info, test_info, train_env_name_list, test_env_name_list, return_info)
    
    #exit()

    ######
    # construct dt model and trainer
    ######
    num_train_env = len(train_env_name_list)
    
    # create model
    state_dim = test_env_list[0].observation_space.shape[0]
    act_dim = test_env_list[0].action_space.shape[0]
    goal_dim = test_info[test_env_name_list[0]]['goal'].shape[0]

    model = PromptDecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        goal_dim=goal_dim,
        max_length=int(variant['K']),
        max_ep_len=1000,
        hidden_size=variant['embed_dim'],
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4 * variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
        no_prompt=variant['no_prompt'],
        prompt_method=variant['prompt_method'],
    )
    model = model.to(device=device)

    print("======> Model created")

    # create optimizer
    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    # create trainer
    if variant['prompt_method'] == "traj_prompt":
        get_prompt_batch_fn = get_prompt_batch(train_trajectories_list, train_prompt_trajectories_list, train_info, variant, train_env_name_list)
        get_prompt_fn = get_prompt
    elif variant['prompt_method'] == "goal_prompt":
        get_prompt_batch_fn = get_goal_prompt_batch(train_trajectories_list, train_info, variant, train_env_name_list)
        get_prompt_fn = get_goal_prompt
    else:
        print("Error: undefined prompt method")
        exit()
    

    trainer = PromptSequenceTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn_type = variant['loss_fn'],
        scheduler=scheduler,
        eval_fns=None,
        get_prompt_batch_fn=get_prompt_batch_fn,   
    )

    print("======> Trainer created")

    if mode == 'train':
        ######
        # start training
        ######

        # set experiment name
        if variant['no_prompt']:
            prompt_method = 'no_prompt'
        else:
            prompt_method = variant['prompt_method']
        
        suffix = ""
        if variant['loss_fn'] == 'predict_rtg':
            suffix += "-pred_rtg"
        elif variant['loss_fn'] == 'predict_reward':
            suffix += "-pred_reward"
        
        if variant['prompt_method'] == "traj_prompt":
            if variant['traj_prompt']['crop_method'] == 'random_crop':
                suffix += "-random_crop"
            elif variant['traj_prompt']['crop_method'] == 'last_step':
                suffix += "-last_step"
        
        suffix += variant['suffix']

        group_name = f'{base_env}-{str(num_train_env)}-env-{train_dataset_mode}-{prompt_method}' + suffix
        now = datetime.datetime.now()
        experiment_name = "s%d-"%(seed) + now.strftime("%Y%m%d-%H%M%S").lower() 

        # set checkpoint folder path
        checkpoint_path = os.path.join(runs_path, group_name + "-" + experiment_name)

        # set evaluation results folder path
        eval_results_path = os.path.join(evaluation_path, group_name + "-" + experiment_name)

        # save config file (save before training to avoid any change of the config file)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        save_config_file_path = os.path.join(checkpoint_path, config_filename)
        shutil.copy(config_file_path, save_config_file_path)
        print("======> Config saved to ", save_config_file_path)
        
        
        # wandb initialize
        if log_to_wandb:
            wandb.init(
                name=experiment_name,
                group=group_name,
                project=project_name,
                config=variant,
                dir=root_path,
            )

        print("======> Start training ...")
        max_iters = variant['max_iters']
        for iter in range(max_iters):
            # each iteration picks a training environment in turn
            env_id = iter % num_train_env
            env_name = train_env_name_list[env_id]

            # train for n update steps
            outputs = trainer.pure_train_iteration_mix(
                num_steps=variant['num_steps_per_iter'], 
                no_prompt=variant['no_prompt']
                )
            
            print("======> Iteration %d train done."%(iter+1))

            # evaluate in test environments (including the first and last iteration)
            if iter % variant['test_eval_interval'] == 0 or iter == max_iters-1:
                test_eval_logs, test_eval_results = trainer.eval_iteration_multienv(
                    get_prompt_fn, test_prompt_trajectories_list,
                    eval_episodes, test_env_name_list, test_info, variant, 
                    test_env_list, iter_num=iter + 1, 
                    print_logs=True, no_prompt=variant['no_prompt'], group='test')
                # update logs
                outputs.update(test_eval_logs)
                # save evaluation results
                save_eval_results(eval_results=test_eval_results, 
                                file_name='iter-'+str(iter)+'.pkl', # iteration index starts from 0
                                folder=os.path.join(eval_results_path, "test"))
                
            # evaluate in train environments (including the first and last iteration)
            if iter % variant['train_eval_interval'] == 0 or iter == max_iters-1:
                train_eval_logs, train_eval_results = trainer.eval_iteration_multienv(
                    get_prompt_fn, train_prompt_trajectories_list,
                    eval_episodes, train_env_name_list, train_info, variant, 
                    train_env_list, iter_num=iter + 1, 
                    print_logs=True, no_prompt=variant['no_prompt'], group='train')
                
                # update logs
                outputs.update(train_eval_logs)
                # save evaluation results
                save_eval_results(eval_results=train_eval_results, 
                                file_name='iter-'+str(iter)+'.pkl', # iteration index starts from 0
                                folder=os.path.join(eval_results_path, "train"))

            # save model (including the first and last iteration)
            if iter % variant['save_interval'] == 0 or iter == max_iters-1:
                trainer.save_model(
                    model_name='iter-'+str(iter)+".pth", # iteration index starts from 0
                    folder=checkpoint_path)

            outputs.update({"global_step": iter}) # set global step as iteration

            # log
            if log_to_wandb:
                wandb.log(outputs)


    else:
        ####
        # start evaluating
        ####

        # load model
        load_path = variant['load_path']
        checkpoint_name = variant['checkpoint_name']
        saved_model_path = os.path.join(runs_path, load_path, checkpoint_name)
        model.load_state_dict(torch.load(saved_model_path))
        print('======> Model loaded from: ', saved_model_path)

        # remove '.pth' from checkpoint_name
        checkpoint_name = checkpoint_name.split('.')[0] 
        eval_iter_num = int(checkpoint_name.split('-')[-1])

        eval_results_path = os.path.join(evaluation_path, load_path)

        # evaluate in test environments
        test_eval_logs, test_eval_results = trainer.eval_iteration_multienv(
                    get_prompt_fn, test_prompt_trajectories_list,
                    eval_episodes, test_env_name_list, test_info, variant, 
                    test_env_list, iter_num=eval_iter_num, 
                    print_logs=True, no_prompt=variant['no_prompt'], group='test')
        
        # save evaluation results
        save_eval_results(eval_results=test_eval_results, 
                        file_name=checkpoint_name+'.pkl', # iteration index starts from 0
                        folder=os.path.join(eval_results_path, "test"))
        
    
        # evaluate in train environments
        train_eval_logs, train_eval_results = trainer.eval_iteration_multienv(
                    get_prompt_fn, train_prompt_trajectories_list,
                    eval_episodes, train_env_name_list, train_info, variant, 
                    train_env_list, iter_num=eval_iter_num, 
                    print_logs=True, no_prompt=variant['no_prompt'], group='train')
    
        # save evaluation results
        save_eval_results(eval_results=train_eval_results, 
                        file_name=checkpoint_name+'.pkl', # iteration index starts from 0
                        folder=os.path.join(eval_results_path, "train"))


        
if __name__ == '__main__':
    #experiment_mix_env(config_filename="cheetah_dir.yaml", mode="train") # mode: ['train', 'eval']
    #experiment_mix_env(config_filename="cheetah_vel.yaml", mode="train")
    #experiment_mix_env(config_filename="ant_dir.yaml", mode="train")
    experiment_mix_env(config_filename="ML1-pick-place-v2.yaml", mode="train")
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

from prompt_dt.prompt_decision_transformer import PromptDecisionTransformer
from prompt_dt.prompt_seq_trainer import PromptSequenceTrainer
from prompt_dt.prompt_utils import get_env_list
from prompt_dt.prompt_utils import get_prompt_batch, get_prompt, get_batch, get_batch_finetune
from prompt_dt.prompt_utils import get_total_data_mean_std, load_data_prompt, process_info
from prompt_dt.prompt_utils import eval_episodes, load_train_test_env_name_list, get_total_num_trajectory
from prompt_dt.utils.path import *
from prompt_dt.utils.other import parse_config, seed_other

from collections import namedtuple
import json, pickle, os


def experiment_mix_env(variant, mode):
    # print out variant
    print("=========================== variant ==================================")
    print(variant)

    # algorithm name
    algorithm_name = variant['algorithm_name']
    
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

    # training envs
    train_info, train_env_list = get_env_list(train_env_name_list, task_config_path, device, seed)
    
    # test envs
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
    # process train dataset info
    train_info = process_info(train_env_name_list, train_trajectories_list, train_info, reward_mode, train_dataset_mode, pct_traj, variant)
    # process test dataset info
    test_info = process_info(test_env_name_list, test_trajectories_list, test_info, reward_mode, test_dataset_mode, pct_traj, variant)

    print("======> Processed train and test trajectories")

    ######
    # construct dt model and trainer
    ######
    num_train_env = len(train_env_name_list)
    
    # create model
    state_dim = test_env_list[0].observation_space.shape[0]
    act_dim = test_env_list[0].action_space.shape[0]

    model = PromptDecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
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
    trainer = PromptSequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=int(variant['batch_size']),
        get_batch=get_batch,
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
        eval_fns=None,
        get_prompt=get_prompt(train_prompt_trajectories_list[0], train_info[train_env_name_list[0]], variant),
        get_prompt_batch=get_prompt_batch(train_trajectories_list, train_prompt_trajectories_list, train_info, variant, train_env_name_list)
    )

    print("======> Trainer created")

    if mode == 'train':
        ######
        # start training
        ######

        # set experiment name
        group_name = f'{base_env}-{str(num_train_env)}-env-{train_dataset_mode}'
        now = datetime.datetime.now()
        experiment_name = "s%d-"%(seed) + now.strftime("%Y%m%d-%H%M%S").lower() 

        # get checkpoint folder path
        checkpoint_path = os.path.join(runs_path, group_name + "-" + experiment_name)
        
        # wandb initialize
        if log_to_wandb:
            wandb.init(
                name=experiment_name,
                group=group_name,
                project=algorithm_name,
                config=variant
            )

        # construct model post fix
        model_post_fix = '_TRAIN_'+variant['train_prompt_mode']+'_TEST_'+variant['test_prompt_mode']
        if variant['no_prompt']:
            model_post_fix += '_NO_PROMPT'
        if variant['finetune']:
            model_post_fix += '_FINETUNE'
        if variant['no_r']:
            model_post_fix += '_NO_R'
        
        print("======> Start training ...")
        for iter in range(variant['max_iters']):
            # each iteration pick an training environment in turn
            env_id = iter % num_train_env
            env_name = train_env_name_list[env_id]

            # train for n update steps
            outputs = trainer.pure_train_iteration_mix(
                num_steps=variant['num_steps_per_iter'], 
                no_prompt=variant['no_prompt']
                )
            
            print("======> Iteration %d train done."%(iter+1))

            # evaluate in test environments
            if iter % variant['test_eval_interval'] == 0:
                if not variant['finetune']:
                    test_eval_logs = trainer.eval_iteration_multienv(
                        get_prompt, test_prompt_trajectories_list,
                        eval_episodes, test_env_name_list, test_info, variant, test_env_list, iter_num=iter + 1, 
                        print_logs=True, no_prompt=variant['no_prompt'], group='test')
                    outputs.update(test_eval_logs)
                else:
                    test_eval_logs = trainer.finetune_eval_iteration_multienv(
                        get_prompt, get_batch_finetune, test_prompt_trajectories_list, test_trajectories_list,
                        eval_episodes, test_env_name_list, test_info, 
                        variant, test_env_list, iter_num=iter + 1, 
                        print_logs=True, no_prompt=variant['no_prompt'], 
                        group='finetune-test', finetune_opt=variant['finetune_opt'])
                    outputs.update(test_eval_logs)
            
            # evaluate in train environments
            if iter % variant['train_eval_interval'] == 0:
                train_eval_logs = trainer.eval_iteration_multienv(
                    get_prompt, train_prompt_trajectories_list,
                    eval_episodes, train_env_name_list, train_info, variant, train_env_list, iter_num=iter + 1, 
                    print_logs=True, no_prompt=variant['no_prompt'], group='train')
                outputs.update(train_eval_logs)

            # save model
            if iter % variant['save_interval'] == 0:
                trainer.save_model(
                    env_name=env_name, 
                    postfix=model_post_fix+'_iter-'+str(iter), # iteration index starts from 0
                    folder=checkpoint_path)

            outputs.update({"global_step": iter}) # set global step as iteration

            # log
            if log_to_wandb:
                wandb.log(outputs)
        
        # save model after the last iteration
        trainer.save_model(env_name=env_name,  
                           postfix=model_post_fix+'_iter_'+str(iter), # iteration index starts from 0 
                           folder=checkpoint_path)
    else:
        ####
        # start evaluating
        ####

        # load model
        saved_model_path = os.path.join(runs_path, variant['load_path'])
        model.load_state_dict(torch.load(saved_model_path))
        print('======> Model loaded from: ', saved_model_path)

        eval_iter_num = int(saved_model_path.split('_')[-1])

        eval_logs = trainer.eval_iteration_multienv(
                    get_prompt, test_prompt_trajectories_list,
                    eval_episodes, test_env_name_list, test_info, variant, test_env_list, iter_num=eval_iter_num, 
                    print_logs=True, no_prompt=variant['no_prompt'], group='eval')

        
if __name__ == '__main__':
    config = parse_config(os.path.join(config_path, "ant_dir.yaml"))
    experiment_mix_env(variant=config, mode="train") # mode: ['train', 'eval']
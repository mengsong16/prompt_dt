# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import numpy as np
import torch
import time
from wandb import env
from prompt_dt.prompt_utils import flatten_prompt
import copy
import os


class PromptSequenceTrainer:

    def __init__(self, model, optimizer, loss_fn,
                 scheduler=None, eval_fns=None, get_prompt_batch_fn=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns

        # get_prompt_batch_fn = get_prompt_batch(train_trajectories_list, train_prompt_trajectories_list, train_info, variant, train_env_name_list)
        self.get_prompt_batch_fn = get_prompt_batch_fn # function with parameters

    # train for one iteration
    def pure_train_iteration_mix(self, num_steps, no_prompt):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            train_loss = self.train_step_mix(no_prompt)
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        return logs

    # train for one step
    def train_step_mix(self, no_prompt):
        # get trajectory prompt batch and trajectory batch
        prompt, batch = self.get_prompt_batch_fn()
        states, actions, rewards, dones, rtg, timesteps, attention_mask = batch
        action_target = torch.clone(actions)
        
        # Note that 
        # states.shape: [B, segment_length, state_dim]
        # rtg.shape: [B, segment_length+1, 1]
        # rtg[:,:-1].shape: [B, segment_length, 1]

        if no_prompt:
            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask, prompt=None
            )
        else:
            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask, prompt=prompt
            )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25) # grad_clip = 0.25 in original dt paper

        self.optimizer.step()

        return loss.detach().cpu().item()


    def eval_iteration_multienv(self, get_prompt_fn, prompt_trajectories_list, 
                                eval_episodes, env_name_list, info, 
                                variant, env_list, iter_num, 
                                print_logs, no_prompt, group):

        print('======> Evaluate at tasks: ', env_name_list)

        logs = dict()
        self.model.eval()

        eval_start = time.time()
        for env_id, env_name in enumerate(env_name_list):
            # set up eval_fns and their parameters
            self.eval_fns = [eval_episodes(tar, info[env_name], variant, env_list[env_id], env_name) for tar in info[env_name]['env_targets']]
            
            if not no_prompt:
                # set up get one prompt fn and its parameters
                current_get_prompt_fn = get_prompt_fn(prompt_trajectories_list[env_id], info[env_name], variant)
                # get a single prompt since we evalute one episode at one time: [number_segments, segment_length, state_dim]
                # concatenate its prompt segments into: [1, prompt_length, state_dim]
                current_prompt = flatten_prompt(current_get_prompt_fn(), batch_size=1)
            else:
                current_prompt = None
            
            # evaluate in current environment for num_eval_episodes
            for eval_fn in self.eval_fns:
                # get return mean and std
                outputs = eval_fn(self.model, prompt=current_prompt)
                for k, v in outputs.items():
                    logs[f'{group}-evaluation/{k}'] = v

        logs['time/evaluation'] = time.time() - eval_start

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')
            print('=' * 80)

        return logs

 
    def save_model(self, model_name, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

        save_path = os.path.join(folder, model_name)
        torch.save(self.model.state_dict(), save_path)
        print('======> Model saved to ', save_path)

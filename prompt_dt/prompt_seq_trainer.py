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

    def __init__(self, model, optimizer, batch_size, get_batch_fn, loss_fn,
                 scheduler=None, eval_fns=None, get_prompt_fn=None, get_prompt_batch_fn=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch_fn = get_batch_fn # function with no parameters
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.get_prompt_fn = get_prompt_fn # function with parameters
        self.prompt = self.get_prompt_fn() # sample a single prompt when initialization
        # get_prompt_batch = get_prompt_batch(train_trajectories_list, train_prompt_trajectories_list, train_info, variant, train_env_name_list)
        self.get_prompt_batch_fn = get_prompt_batch_fn # function with parameters

        self.start_time = time.time()

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

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

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

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()


    def finetune_eval_iteration_multienv(self, get_prompt_fn, get_batch_fn, 
                                test_prompt_trajectories_list, test_trajectories_list, 
                                eval_episodes, env_name_list, info, 
                                variant, env_list, iter_num=0, print_logs=False, 
                                no_prompt=False, group='test-finetune',
                                finetune_opt=False):
        print('======> Evaluate at tasks: ', env_name_list)

        logs = dict()
        self.model.eval()

        # model before finetune
        self.current_model_dict = copy.deepcopy(self.model.state_dict())

        eval_start = time.time()

        # set up finetune optimizer
        if finetune_opt:
            fintune_optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=variant['finetune_lr'],
                weight_decay=1e-4,
            )
        else:
            fintune_optimizer = None
        
        for env_id, env_name in enumerate(env_name_list):
            # set up eval_fns and their parameters
            self.eval_fns = [eval_episodes(tar, info[env_name], variant, env_list[env_id], env_name) for tar in info[env_name]['env_targets']]
            # set up get_batch function and its parameters
            self.get_batch_fn = get_batch_fn(test_trajectories_list[env_id], info[env_name], variant)
            if not no_prompt:
                # set up get one prompt fn and its parameters
                self.get_prompt_fn = get_prompt_fn(test_prompt_trajectories_list[env_id], info[env_name], variant)
                # get a single prompt since we evalute one episode at one time: [number_segments, segment_length, state_dim]
                # this also means that we use one prompt for the whole finetune batch
                # concatenate its prompt segments into: [1, prompt_length, state_dim]
                self.prompt = flatten_prompt(self.get_prompt_fn(), batch_size=1) 
            else:
                self.prompt = None

            # finetune the model on the test trajectories for current test environment
            self.model.train()
            finetune_losses = []
            for _ in range(variant['finetune_steps']):
                finetune_loss = self.finetune_step(
                    batch_size_overwrite=variant['finetune_batch_size'],
                    optimizer=fintune_optimizer)
                finetune_losses.append(finetune_loss)

            # evaluation
            self.model.eval()

            for eval_fn in self.eval_fns:
                outputs = eval_fn(self.model, prompt=self.prompt)
                for k, v in outputs.items():
                    logs[f'{group}-evaluation/{k}'] = v
            
            # recover to the model before finetune
            # only finetune in the current test environment, evaluate then remove the effect of finetune
            self.model.load_state_dict(self.current_model_dict)

        logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')
            
            print('=' * 80)

        return logs

    # finetune (continue train) for one step
    def finetune_step(self, batch_size_overwrite, optimizer=None):
        # use given batch_size instead of batch size for regular training
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch_fn(batch_size_overwrite)
        
        action_target = torch.clone(actions)

        # Note that 
        # states.shape: [B, segment_length, state_dim]
        # rtg.shape: [B, segment_length+1, 1]
        # rtg[:,:-1].shape: [B, segment_length, 1]
        
        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask, prompt=self.prompt
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        if optimizer is None:
            self.optimizer.zero_grad()
        else:
            optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25) # grad_clip = 0.25 in original dt paper

        if optimizer is None:
            self.optimizer.step()
        else:
            optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()


    def eval_iteration_multienv(self, get_prompt_fn, prompt_trajectories_list, 
                                eval_episodes, env_name_list, info, 
                                variant, env_list, iter_num=0, 
                                print_logs=False, no_prompt=False, group='test'):

        print('======> Evaluate at tasks: ', env_name_list)

        logs = dict()
        self.model.eval()

        eval_start = time.time()
        for env_id, env_name in enumerate(env_name_list):
            # set up eval_fns and their parameters
            self.eval_fns = [eval_episodes(tar, info[env_name], variant, env_list[env_id], env_name) for tar in info[env_name]['env_targets']]
            
            if not no_prompt:
                # set up get one prompt fn and its parameters
                self.get_prompt_fn = get_prompt_fn(prompt_trajectories_list[env_id], info[env_name], variant)
                # get a single prompt since we evalute one episode at one time: [number_segments, segment_length, state_dim]
                # concatenate its prompt segments into: [1, prompt_length, state_dim]
                self.prompt = flatten_prompt(self.get_prompt_fn(), batch_size=1)
            else:
                self.prompt = None
            
            for eval_fn in self.eval_fns:
                outputs = eval_fn(self.model, prompt=self.prompt)
                for k, v in outputs.items():
                    logs[f'{group}-evaluation/{k}'] = v

        logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')
            print('=' * 80)

        return logs

 
    def save_model(self, env_name, postfix, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

        model_name = env_name + postfix
        save_path = os.path.join(folder, model_name)
        torch.save(self.model.state_dict(), save_path)
        print('======> Model saved to ', save_path)

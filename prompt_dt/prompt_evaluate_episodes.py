# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import numpy as np
import torch
import time
import os
import pickle 

""" evaluation """

def eval_episodes(target_rew, info, variant, env, env_name):
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    num_eval_episodes = variant['num_eval_episodes']
    reward_mode = variant.get('reward_mode', 'normal')

    def fn(model, prompt=None):
        returns = []
        episode_lengths = []
        for _ in range(num_eval_episodes):
            with torch.no_grad():
                # evaluate for one episode
                # return episode_return and infos['episode_length']
                ret, infos = prompt_evaluate_episode_rtg(
                    env,
                    state_dim,
                    act_dim,
                    model,
                    max_ep_len=max_ep_len,
                    scale=scale,
                    target_return=target_rew / scale,
                    reward_mode=reward_mode,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                    prompt=prompt,
                    no_r=variant['no_r'],
                    no_rtg=variant['no_rtg'],
                    no_state_normalize=variant['no_state_normalize']                
                    )
            returns.append(ret)
            episode_lengths.append(infos['episode_length'])
        
        return {
            f'{env_name}_target_{target_rew}_return_mean': np.mean(returns),
            f'{env_name}_target_{target_rew}_return_std': np.std(returns),
            }, returns, episode_lengths, target_rew
    return fn

# evaluate policy for one episode in a test environment
# w or w/o prompt
def prompt_evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len,
        scale,
        state_mean,
        state_std,
        device,
        target_return,
        reward_mode,
        prompt, # a single prompt
        no_r,
        no_rtg,
        no_state_normalize
    ):

    model.eval()
    model.to(device=device)

    # for state normalization
    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if reward_mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories of states, rewards, actions, timesteps on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):
        # print('evaluate/t', t)
        
        # initialize action and reward history as a single 0
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        if no_state_normalize:
            action = model.get_action(
                states.to(dtype=torch.float32),
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
                prompt=prompt
            )
        else:
            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
                prompt=prompt
            )
            
        # append new action to the rightmost location of the action sequence
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, infos = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        # append new state to the rightmost location of the state history
        states = torch.cat([states, cur_state], dim=0)
        # append new reward to the rightmost location of the reward history
        rewards[-1] = reward
        if no_r:
            rewards[-1] = 0.0

        # append new rtg (pred_return) to the rightmost location of the rtg history
        # target_return[0,-1] is the current last rtg in the history
        # if delayed, the rtg history will be a sequence with constant rtg until the episode ends
        if reward_mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]

        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        # if no return-to-go, use a constant target return
        if no_rtg:
            target_return = torch.ones_like(target_return)*ep_return
        # append new timestep to the rightmost location of the timestep history
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        infos['episode_length'] = episode_length

        if done:
            break


    return episode_return, infos

# pack the evaluation results for a given (environment, return_target) pair
# target_return is unscaled
def pack_eval_results_one_env_target(env_id, env_name,
                                    returns, episode_lengths,
                                    target_return):
    eval_results = {}
    eval_results["env_id"] = env_id
    eval_results["env_name"] = env_name
    eval_results["returns"] = returns
    eval_results["episode_lengths"] = episode_lengths
    eval_results["target_return"] = target_return

    return eval_results

# eval_results is a list of results for each (environment, return_target) pair
def compute_mean_std_one_base_env_multi_targets(eval_results):
    returns = {}
    episode_lengths = {}
    base_env_name = eval_results[0]["env_name"].split('-')[0]

    for cur_env_target_results in eval_results:
        cur_target_return = cur_env_target_results["target_return"]
        # returns
        if cur_target_return not in returns:
            returns[cur_target_return] = []
        else:
            returns[cur_target_return].extend(cur_env_target_results["returns"])
        
        # episode_lengths
        if cur_target_return not in episode_lengths:
            episode_lengths[cur_target_return] = []
        else:
            episode_lengths[cur_target_return].extend(cur_env_target_results["episode_lengths"])

    eval_stats = {}
    for target_return, rts in returns.items():
        rts = np.array(rts)
        eval_stats[f'{base_env_name}_target_{target_return}_return_mean'] = np.mean(rts)
        eval_stats[f'{base_env_name}_target_{target_return}_return_std'] = np.std(rts)
    
    for target_return, epi_lens in episode_lengths.items():
        epi_lens = np.array(epi_lens)
        eval_stats[f'{base_env_name}_target_{target_return}_episode_length_mean'] = np.mean(epi_lens)
        eval_stats[f'{base_env_name}_target_{target_return}_episode_length_std'] = np.std(epi_lens)

    return eval_stats

def save_eval_results(eval_results, file_name, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    save_path = os.path.join(folder, file_name)
    
    with open(save_path, 'wb') as f:
        pickle.dump(eval_results, f)

    print('======> Evaluation results saved to ', save_path)

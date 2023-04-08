# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import numpy as np
import torch
import time
import os
import pickle 
from prompt_dt.prompt_utils import flatten_trajectory_prompt
#from gym.wrappers.monitoring.video_recorder import VideoRecorder

""" evaluation """
# evaluate policy with a given target rtg for n episodes in a test environment
def eval_episodes(target_rew, info, variant, env, env_name, render=False):
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    num_eval_episodes = variant['num_eval_episodes']
    reward_mode = variant.get('reward_mode', 'normal')


    def fn(model, prompt=None):
        returns = []
        episode_lengths = []
        # evaluate for n episodes with the same prompt
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
                    no_state_normalize=variant['no_state_normalize'],
                    render=render              
                    )
            returns.append(ret)
            episode_lengths.append(infos['episode_length'])


        return {
            f'{env_name}_target_{int(target_rew)}_return_mean': np.mean(returns),
            f'{env_name}_target_{int(target_rew)}_return_std': np.std(returns),
            }, returns, episode_lengths, target_rew
    return fn

# evaluate policy with a given target rtg for one episode in a test environment
# w or w/o prompt
# here input target_return has been divided by scale
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
        no_state_normalize,
        render
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

        if render:
            env.render()

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        # append new state to the rightmost location of the state history
        states = torch.cat([states, cur_state], dim=0)
        # append new reward to the rightmost location of the reward history
        rewards[-1] = reward
        # if no reward in input trajectories, use 0 as reward
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
        # if no return-to-go in input trajectories, use a constant target return
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

# pack the evaluation results for a given (environment, return_target) pair into a dictionary
# returns is a list
# episode_lengths is a list
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

# eval_results is a list of results 
# each results is for each (environment, return_target) pair
# all environments are from the same base_env
def compute_mean_std_one_base_env_multi_targets(eval_results):
    returns = {}
    episode_lengths = {}
    base_env_name = eval_results[0]["env_name"].split('-')[0]

    # returns is a dictionary: group evaluation returns by target rtg across all environments
    # episode_lengths is a dictionary: group evaluation episode lengths by target rtg across all environments
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
    # compute return mean/std and episode length mean/std for reach target rtg
    for target_return, rts in returns.items():
        rts = np.array(rts)
        eval_stats[f'{base_env_name}_target_{target_return}_return_mean'] = np.mean(rts)
        eval_stats[f'{base_env_name}_target_{target_return}_return_std'] = np.std(rts)
    
    for target_return, epi_lens in episode_lengths.items():
        epi_lens = np.array(epi_lens)
        eval_stats[f'{base_env_name}_target_{target_return}_episode_length_mean'] = np.mean(epi_lens)
        eval_stats[f'{base_env_name}_target_{target_return}_episode_length_std'] = np.std(epi_lens)

    return eval_stats

# normalize a single return
def normalize_one_return(ret, expert_return, random_return):
    normalized_return = 100.0 * (ret - random_return) / (expert_return - random_return)
    return normalized_return

# normalize a list of returns
def normalize_returns(returns, expert_return, random_return):
    normalized_returns = []
    for ret in returns:
        normalized_return = normalize_one_return(ret, expert_return, random_return)
        normalized_returns.append(normalized_return)

    return normalized_returns

# eval_results are from the same base env
def compute_episode_length_normalized_score(eval_results, info):
    base_env_name = eval_results[0]["env_name"].split('-')[0]
    all_episode_lengths = []
    all_normalized_returns = []
    for cur_env_target_results in eval_results:
        env_name = cur_env_target_results["env_name"]
        # get evaluation returns and episode lengths for current (env, target_rtg) pair
        returns = cur_env_target_results["returns"]
        # no need to normalize episode length since envs from the same base env has the same maximum episode length
        episode_lengths = cur_env_target_results["episode_lengths"]
        # normalize return to [0, 100]
        expert_return = info[env_name]['max_return']
        random_return = info[env_name]['random_return']
        normalized_returns = normalize_returns(returns, expert_return, random_return)
        # collect data
        all_episode_lengths.extend(episode_lengths)
        all_normalized_returns.extend(normalized_returns)
    
    # compute return mean/std and episode length mean/std
    eval_stats = {}
    all_episode_lengths = np.array(all_episode_lengths)
    all_normalized_returns = np.array(all_normalized_returns)

    eval_stats[f'{base_env_name}_return_mean'] = np.mean(all_normalized_returns)
    eval_stats[f'{base_env_name}_return_std'] = np.std(all_normalized_returns)
    eval_stats[f'{base_env_name}_episode_length_mean'] = np.mean(all_episode_lengths)
    eval_stats[f'{base_env_name}_episode_length_std'] = np.std(all_episode_lengths)

    return eval_stats


def save_eval_results(eval_results, file_name, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    save_path = os.path.join(folder, file_name)
    
    with open(save_path, 'wb') as f:
        pickle.dump(eval_results, f)

    print('======> Evaluation results saved to ', save_path)

# compute prompt in a given environment during evaluation
def get_prompt_eval(env_id, env_name,
                get_prompt_fn,
                variant, prompt_trajectories_list, info):
    if variant['prompt_method'] == "traj_prompt":
        # set up get one prompt fn and its parameters
        current_get_prompt_fn = get_prompt_fn(prompt_trajectories_list[env_id], info[env_name], variant)
        # get a single trajectory prompt since we evalute one episode at one time: [number_segments, segment_length, state_dim]
        # concatenate its prompt segments info: [1, prompt_length, state_dim]
        # Note that rtg's sequence length has been decreased 1 to the correct length in flatten_trajectory_prompt
        current_prompt = flatten_trajectory_prompt(current_get_prompt_fn(), batch_size=1)
    elif variant['prompt_method'] == "goal_prompt" or variant['prompt_method'] == "goal_learned_prompt":
        # get a single goal prompt 
        current_prompt = get_prompt_fn(info[env_name])
    elif variant["prompt_method"] == "goal_state_prompt": 
        # set up get one prompt fn and its parameters
        current_get_prompt_fn = get_prompt_fn(prompt_trajectories_list[env_id], info[env_name])
        # get a single goal state prompt
        current_prompt = current_get_prompt_fn()
    elif variant['prompt_method'] == "no_prompt" or variant['prompt_method'] == "pure_learned_prompt":
        # prompt is None
        current_prompt = get_prompt_fn()
    else:
        print("Error: Unknown prompt method")
        exit()
    
    
    return current_prompt
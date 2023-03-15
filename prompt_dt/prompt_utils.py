import numpy as np
import gym
import json, pickle, random, os, torch
from collections import namedtuple
from prompt_dt.prompt_evaluate_episodes import prompt_evaluate_episode_rtg
from prompt_dt.utils.path import *

# for mujoco tasks
from mujoco_control_envs.mujoco_control_envs import HalfCheetahDirEnv, HalfCheetahVelEnv, AntDirEnv
# for jacopinpad
from jacopinpad.jacopinpad_gym import jacopinpad_multi
# for metaworld
import metaworld
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)

""" constructing envs """
def load_train_test_env_name_list(env_name):
    task_config = os.path.join(task_config_path, config_path_dict[env_name])
    with open(task_config, 'r') as f:
        task_config = json.load(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    train_env_name_list, test_env_name_list = [], []
    for task_ind in task_config.train_tasks:
        train_env_name_list.append(env_name +'-'+ str(task_ind))
    for task_ind in task_config.test_tasks:
        test_env_name_list.append(env_name +'-'+ str(task_ind))
    
    print("======================== traing envs:%d =============================="%(len(train_env_name_list)))
    print(train_env_name_list)
    print("======================== test envs: %d ================================"%(len(test_env_name_list)))
    print(test_env_name_list)
    
    return train_env_name_list, test_env_name_list

# create a single environment with reward scale, maximum episode length, return target
def gen_env(env_name, config_save_path, seed):
    if 'cheetah_dir' in env_name:
        # include_goal = False: do not include goal in observations
        if '0' in env_name:  # direction 1
            env = HalfCheetahDirEnv([{'direction': 1}], include_goal = False)
        elif '1' in env_name: # direction -1
            env = HalfCheetahDirEnv([{'direction': -1}], include_goal = False)
        max_ep_len = 200
        env_targets = [1000] #[1500]
        scale = 1000.
    elif 'cheetah_vel' in env_name:
        task_idx = int(env_name.split('-')[-1])
        task_paths = f"{config_save_path}/cheetah_vel/config_cheetah_vel_task{task_idx}.pkl"
        tasks = []
        with open(task_paths.format(task_idx), 'rb') as f:
            task_info = pickle.load(f)
            assert len(task_info) == 1, f'Unexpected task info: {task_info}'
            tasks.append(task_info[0])
        # include_goal = False: do not include goal in observations
        env = HalfCheetahVelEnv(tasks, include_goal = False)
        max_ep_len = 200
        env_targets = [0]
        scale = 500.
    elif 'ant_dir' in env_name:
        task_idx = int(env_name.split('-')[-1])
        task_paths = f"{config_save_path}/ant_dir/config_ant_dir_task{task_idx}.pkl"
        tasks = []
        with open(task_paths.format(task_idx), 'rb') as f:
            task_info = pickle.load(f)
            assert len(task_info) == 1, f'Unexpected task info: {task_info}'
            tasks.append(task_info[0])
        # include_goal = False: do not include goal in observations
        env = AntDirEnv(tasks, len(tasks), include_goal = False)
        max_ep_len = 200
        env_targets = [500]
        scale = 500.
    elif 'ML1-' in env_name: # metaworld ML1
        task_name = '-'.join(env_name.split('-')[1:-1])
        ml1 = metaworld.ML1(task_name, seed=seed) # construct the benchmark, sampling tasks
        env = ml1.train_classes[task_name]()  # create an environment with task
        task_idx = int(env_name.split('-')[-1])
        task = ml1.train_tasks[task_idx]
        env.set_task(task)  # set task
        max_ep_len = 500 
        env_targets= [650]
        scale = 650.
    else:
        raise NotImplementedError
    return env, max_ep_len, env_targets, scale

# load a list of environments
# pack environment info
def get_env_list(env_name_list, config_save_path, device, seed):
    info = {} # store all the attributes for each env
    env_list = []
    
    for env_name in env_name_list:
        info[env_name] = {}
        env, max_ep_len, env_targets, scale = gen_env(env_name=env_name, config_save_path=config_save_path, seed=seed)
        info[env_name]['max_ep_len'] = max_ep_len
        info[env_name]['env_targets'] = env_targets
        info[env_name]['scale'] = scale
        info[env_name]['state_dim'] = env.observation_space.shape[0]
        info[env_name]['act_dim'] = env.action_space.shape[0] 
        info[env_name]['device'] = device
        env_list.append(env)
    return info, env_list

# count total number of trajectories in a dictionary
def get_total_num_trajectory(trajectory_num):
    total_num = 0
    for n in trajectory_num.values():
        total_num += n
    
    return total_num

""" trajectory prompts """
# reshape trajectory prompts to a new batch_size
# [old_batch_size, segment_length, state_dim] --> [new_batch_size, -1, state_dim]
# -1: old_batch_size * segment_length / new_batch_size
def flatten_prompt(prompt, batch_size):
    p_s, p_a, p_r, p_d, p_rtg, p_timesteps, p_mask = prompt
    # p_s: [old_batch_size, segment_length, state_dim]=[16, 5, 27]
    # if new_batch_size == old_batch_size, nothing changes
    # if new_batch_size == 1 --> concatenate all segments into one
    # ---> [1, old_batch_size * segment_length, state_dim]
    
    p_s = p_s.reshape((batch_size, -1, p_s.shape[-1]))
    p_a = p_a.reshape((batch_size, -1, p_a.shape[-1]))
    p_r = p_r.reshape((batch_size, -1, p_r.shape[-1]))
    p_d = p_d.reshape((batch_size, -1))
    p_rtg = p_rtg[:,:-1,:]
    p_rtg = p_rtg.reshape((batch_size, -1, p_rtg.shape[-1]))
    p_timesteps = p_timesteps.reshape((batch_size, -1))
    p_mask = p_mask.reshape((batch_size, -1)) 
    return p_s, p_a, p_r, p_d, p_rtg, p_timesteps, p_mask

# get one trajectory prompt
def get_prompt(prompt_trajectories, info, variant):
    num_trajectories, p_sample, sorted_inds = info['num_trajectories'], info['p_sample'], info['sorted_inds']
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    num_episodes, max_len = variant['traj_prompt']['prompt_episode'], variant['traj_prompt']['prompt_length']

    def fn(sample_size=1):
        # random sample a batch of prompt trajectories from the whole trajectory pool (p_sample=1)
        # batch_size = num_episodes*sample_size = num_episodes
        batch_inds = np.random.choice(
            np.arange(len(prompt_trajectories)),
            size=int(num_episodes*sample_size),
            replace=True,
        )

        # crop a segement of fixed length (prompt-length) from each prompt trajectory in the batch
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(int(num_episodes*sample_size)):
            if variant["traj_prompt"]["stochastic_prompt"]:
                # randomly select a trajectory from the pool
                traj = prompt_trajectories[int(batch_inds[i])] 
            else:
                # select the trajectory with the return from highest to lowest
                traj = prompt_trajectories[int(sorted_inds[-i])] 

            # si is the beginning of the last segment of length max_len in the trajectory
            si = max(0, traj['rewards'].shape[0] - max_len -1) # select the last part of the traj with length max_len

            # append the segment
            append_new_segment(traj, si, max_len, max_ep_len, 
                       state_dim, act_dim, variant, 
                       state_mean, state_std, scale,
                       s, a, r, d, rtg, timesteps, mask)
            
        # numpy to torch tensor
        s, a, r, d, rtg, timesteps, mask = numpy_to_tensor(s, a, r, d, rtg, timesteps, mask, device)
        
        return s, a, r, d, rtg, timesteps, mask

    return fn

# get a batch of trajectories and trajectory prompts
# Note that for each environment, collect a batch of regular trajectories 
# and a batch of trajectory prompts with batch size per_env_batch_size
def get_prompt_batch(trajectories_list, prompt_trajectories_list, info, variant, train_env_name_list):
    per_env_batch_size = variant['batch_size']

    # Note that batch_size=per_env_batch_size
    def fn(batch_size=per_env_batch_size):
        p_s_list, p_a_list, p_r_list, p_d_list, p_rtg_list, p_timesteps_list, p_mask_list = [], [], [], [], [], [], []
        s_list, a_list, r_list, d_list, rtg_list, timesteps_list, mask_list = [], [], [], [], [], [], []
        for env_id, env_name in enumerate(train_env_name_list):
            # set up get prompt function
            # crop trajectory prompt from prompt trajectories
            if prompt_trajectories_list:
                get_prompt_fn = get_prompt(prompt_trajectories_list[env_id], info[env_name], variant)
            # crop trajectory prompt from regular trajectories
            else:
                get_prompt_fn = get_prompt(trajectories_list[env_id], info[env_name], variant)
            
            # set up get batch function
            get_batch_fn = get_batch(trajectories_list[env_id], info[env_name], variant) 
            
            # get a batch of trajectory prompts
            prompt = flatten_prompt(get_prompt_fn(batch_size), batch_size) # flatten_prompt changes nothing
            p_s, p_a, p_r, p_d, p_rtg, p_timesteps, p_mask = prompt
            p_s_list.append(p_s)
            p_a_list.append(p_a)
            p_r_list.append(p_r)
            p_d_list.append(p_d)
            p_rtg_list.append(p_rtg)
            p_timesteps_list.append(p_timesteps)
            p_mask_list.append(p_mask)

            # get a batch of regular trajectories
            batch = get_batch_fn(batch_size=batch_size)
            s, a, r, d, rtg, timesteps, mask = batch
            if variant['no_r']:
                r = torch.zeros_like(r)
            if variant['no_rtg']:
                rtg = torch.zeros_like(rtg)
            s_list.append(s)
            a_list.append(a)
            r_list.append(r)
            d_list.append(d)
            rtg_list.append(rtg)
            timesteps_list.append(timesteps)
            mask_list.append(mask)

        # from numpy to tensor
        p_s, p_a, p_r, p_d = torch.cat(p_s_list, dim=0), torch.cat(p_a_list, dim=0), torch.cat(p_r_list, dim=0), torch.cat(p_d_list, dim=0)
        p_rtg, p_timesteps, p_mask = torch.cat(p_rtg_list, dim=0), torch.cat(p_timesteps_list, dim=0), torch.cat(p_mask_list, dim=0)
        s, a, r, d = torch.cat(s_list, dim=0), torch.cat(a_list, dim=0), torch.cat(r_list, dim=0), torch.cat(d_list, dim=0)
        rtg, timesteps, mask = torch.cat(rtg_list, dim=0), torch.cat(timesteps_list, dim=0), torch.cat(mask_list, dim=0)
        prompt = p_s, p_a, p_r, p_d, p_rtg, p_timesteps, p_mask
        batch = s, a, r, d, rtg, timesteps, mask

        return prompt, batch
    return fn

""" batches """
def numpy_to_tensor(s, a, r, d, rtg, timesteps, mask, device):
    s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
    a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
    r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
    d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
    rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
    timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
    mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

    return s, a, r, d, rtg, timesteps, mask

def append_new_segment(traj, si, max_len, max_ep_len, 
                       state_dim, act_dim, variant, 
                       state_mean, state_std, scale,
                       s, a, r, d, rtg, timesteps, mask):
    # Note that if si+max_len exceeds current trajectory length, only fetch elements until the episode ends
    s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
    a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
    r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
    if 'terminals' in traj:
        d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
    else:
        d.append(traj['dones'][si:si + max_len].reshape(1, -1))
    # each timestep is the step index inside this segment 
    # index starting from the begining of the trajectory: e.g. [5,6,7]
    # s[-1].shape[1] is the length of current segment (must <= max_len)
    timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
    # if actual index exceed predefined max episode length, use the last step index (i.e. index max_ep_len - 1)
    # timesteps[-1]: current segment
    # timesteps[-1] >= max_ep_len: for each step in current segment, check whether it exceeds max_ep_len
    timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1
    # undiscounted return since gamma = 1
    # first compute each state from si until the episode ends, then cut off at the current segment length + 1
    # s[-1].shape: (1, max_len, state_dim)
    # r[-1].shape: (1, max_len, 1)
    # new_rtg.shape: (1, max_len+1, 1)
    # the extra step in new_rtg will be discarded when batch is used in training
    new_rtg = discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1)
    rtg.append(new_rtg)
    # append a single 0 to rtg
    # this happens when [si, end_of_episode] is shorter than length of s[-1] + 1
    if rtg[-1].shape[1] <= s[-1].shape[1]:  
        rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

    # left pad, state normalization, scale return-to-go
    # tlen is the true length of current segment (<= max_len)
    tlen = s[-1].shape[1]

    # left pad state with 0 if shorter than max_len
    s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
    # normalize state distribution to N(0,1)
    if not variant['no_state_normalize']:
        s[-1] = (s[-1] - state_mean) / state_std
    # left pad action with -10 if shorter than max_len
    a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
    # left pad reward with 0 if shorter than max_len
    # Note that reward is not scaled
    r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
    # left pad done with 2 if shorter than max_len
    d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
    # left pad rtg with 0 if shorter than max_len
    # divide rtg by reward scale
    rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
    # left pad timestep with 0 if shorter than max_len
    timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
    # mask = 1 (attend) until tlen, after that = 0 (not attend)
    mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

# get a batch of regular trajectories
def get_batch(trajectories, info, variant):
    num_trajectories, p_sample, sorted_inds = info['num_trajectories'], info['p_sample'], info['sorted_inds']
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    batch_size, K = variant['batch_size'], variant['K']

    def fn(batch_size=batch_size, max_len=K):
        # randomly sample batch_size trajectories from the top trajectory pool with replacement
        # prefer long trajectory
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to trajectory length
        )

        # crop a segement from each trajectory in the batch
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            # current trajectory
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            # randomly pick a segment of length max_len from current trajectory starting from state si
            si = random.randint(0, traj['rewards'].shape[0] - 1)
            # append the segment
            append_new_segment(traj, si, max_len, max_ep_len, 
                       state_dim, act_dim, variant, 
                       state_mean, state_std, scale,
                       s, a, r, d, rtg, timesteps, mask)
            
        # numpy to torch tensor
        s, a, r, d, rtg, timesteps, mask = numpy_to_tensor(s, a, r, d, rtg, timesteps, mask, device)
        
        return s, a, r, d, rtg, timesteps, mask

    return fn


""" data processing """

def get_total_data_mean_std(trajectories):
    # colect states from all trajectories
    states = []
    for path in trajectories:
        states.append(path['observations'])

    # compute mean and standard deviation over states from all trajectories
    # used for input normalization
    # avoid std=0 by adding 1e-6
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    return state_mean, state_std

# process trajectories from a specific environment
def process_dataset(trajectories, reward_mode, env_name, dataset, pct_traj, verbose):
    # parse all path information into separate lists: states, traj_lens, returns
    # rewrite the reward mode of the trajectories
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if reward_mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for state normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    if verbose:
        print('=' * 50)
        print(f'Processing data from environment: {env_name} {dataset}')
        print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
        print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
        print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
        print('=' * 50)

    # only train/test on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # sort return lowest to highest
    num_trajectories = 1  # the number of top trajectories
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    # sorted_inds: only keep indices of top trajectories
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    
    sorted_inds = sorted_inds[-num_trajectories:]

    # percentage according to trajectory length
    # used to reweight sampling so we sample trajectory according to its length ratio
    # only consider top trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
    # mean/std/max/min of returns
    reward_info = [np.mean(returns), np.std(returns), np.max(returns), np.min(returns)]

    # note that trajectories are still all trajectories (no cut-off), the only difference is the reward mode
    # num_trajectories, sorted_inds, p_sample are for top trajectories
    return trajectories, num_trajectories, sorted_inds, p_sample, state_mean, state_std, reward_info

# load trajectories and prompt trajectories
def load_data_prompt(env_name_list, data_save_path, dataset, prompt_mode, base_env):
    trajectories_list = [] # a list of trajectory list, each trajectory list comes from a specific environment
    prompt_trajectories_list = [] # a list of trajectory list, each trajectory list comes from a specific environment

    trajectory_num = {}
    prompt_trajectory_num = {}
    for env_name in env_name_list:
        dataset_path = data_save_path+f'/{base_env}/{env_name}-{dataset}.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)

        prompt_dataset_path = data_save_path+f'/{base_env}/{env_name}-prompt-{prompt_mode}.pkl'
        with open(prompt_dataset_path, 'rb') as f:
            prompt_trajectories = pickle.load(f)

        trajectories_list.append(trajectories)
        prompt_trajectories_list.append(prompt_trajectories)

        trajectory_num[env_name] = len(trajectories)
        prompt_trajectory_num[env_name] = len(prompt_trajectories)
    
    return trajectories_list, prompt_trajectories_list, trajectory_num, prompt_trajectory_num

# process train/test dataset
def process_info(env_name_list, trajectories_list, info, reward_mode, dataset, pct_traj, variant, verbose=False):
    for i, env_name in enumerate(env_name_list):
        # process trajectories from currect environment
        trajectories, num_trajectories, sorted_inds, p_sample, state_mean, state_std, reward_info = process_dataset(
            trajectories=trajectories_list[i], reward_mode=reward_mode, env_name=env_name_list[i], dataset=dataset, 
            pct_traj=pct_traj, verbose=verbose)
        
        info[env_name]['num_trajectories'] = num_trajectories
        info[env_name]['sorted_inds'] = sorted_inds
        info[env_name]['p_sample'] = p_sample
        info[env_name]['state_mean'] = state_mean
        info[env_name]['state_std'] = state_std

        # change state mean std from a specific environment trajectories to all train+test trajectories
        if variant['average_state_mean']:
            info[env_name]['state_mean'] = variant['total_state_mean']
            info[env_name]['state_std'] = variant['total_state_std']
    
    return info

# compute discounted return at each step in the sequence x
def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum




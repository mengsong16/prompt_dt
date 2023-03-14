# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import numpy as np
import torch
import time

# evaluate policy in a test environment
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

        if done:
            break

        infos['episode_length'] = episode_length

    return episode_return, infos

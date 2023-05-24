import numpy as np
import gym
import random
import json, pickle, random, os, torch
import copy
from task_prompt_dt.prompt_utils import get_env_goal
from task_prompt_dt.utils.other import seed_other
from task_prompt_dt.utils.path import *

# for metaworld
import metaworld
#from metaworld.tests.metaworld.envs.mujoco.sawyer_xyz.utils import trajectory_summary

from metaworld.policies.sawyer_reach_v2_policy import SawyerReachV2Policy
from metaworld.policies.sawyer_push_v2_policy import SawyerPushV2Policy
from metaworld.policies.sawyer_pick_place_v2_policy import SawyerPickPlaceV2Policy
from metaworld.policies.sawyer_door_open_v2_policy import SawyerDoorOpenV2Policy
from metaworld.policies.sawyer_drawer_close_v2_policy import SawyerDrawerCloseV2Policy
from metaworld.policies.sawyer_button_press_topdown_v2_policy import SawyerButtonPressTopdownV2Policy
from metaworld.policies.sawyer_peg_insertion_side_v2_policy import SawyerPegInsertionSideV2Policy
from metaworld.policies.sawyer_window_open_v2_policy import SawyerWindowOpenV2Policy
from metaworld.policies.sawyer_sweep_v2_policy import SawyerSweepV2Policy
from metaworld.policies.sawyer_basketball_v2_policy import SawyerBasketballV2Policy

policies = {
    'reach-v2': SawyerReachV2Policy,
    'push-v2': SawyerPushV2Policy,
    'pick-place-v2': SawyerPickPlaceV2Policy,
    'door-open-v2': SawyerDoorOpenV2Policy,
    'drawer-close-v2': SawyerDrawerCloseV2Policy,
    'button-press-topdown-v2': SawyerButtonPressTopdownV2Policy,
    'peg-insert-side-v2': SawyerPegInsertionSideV2Policy,
    'window-open-v2': SawyerWindowOpenV2Policy,
    'sweep-v2': SawyerSweepV2Policy,
    'basketball-v2': SawyerBasketballV2Policy,
}

def generate_one_subtask(env, policy, env_name, subtask_idx, input_traj_per_subtask, prompt_traj_per_subtask):
    # generate input trajectories for current subtask
    print(f"================= ML1-{env_name}-{subtask_idx}-expert ==================")

    input_trajectories = []

    for traj_ind in range(input_traj_per_subtask): 
        # note that goal_distance, grasp_success is not a general component in info, but 'success'[float] is 
        cur_traj = { 'observations': [], 'actions': [], 'rewards': [], 'terminals': [], 'success': [] }

        obs = env.reset()
        # zero out goal
        zero_out_obs = copy.deepcopy(obs)
        zero_out_obs[-3:] = np.zeros(3)
        cur_traj['observations'].append(zero_out_obs)
        
        for step in range(env.max_path_length+1): 
            # no noise added to the action
            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)
            success = info['success']
            
            # zero out goal
            zero_out_obs = copy.deepcopy(obs)
            zero_out_obs[-3:] = np.zeros(3)

            cur_traj['observations'].append(zero_out_obs)
            cur_traj['actions'].append(action)
            cur_traj['rewards'].append(reward)
            cur_traj['terminals'].append(done) # won't be used
            cur_traj['success'].append(success) # won't be used
        

            if done:
                break
            
        # throw away s_T
        cur_traj['observations'] = cur_traj['observations'][:-1]

        assert len(cur_traj['observations']) == len(cur_traj['actions']) == len(cur_traj['rewards']) == len(cur_traj['terminals']) == len(cur_traj['success']), "All values should have the same length"

        # convert from list to numpy array for each value
        for key in cur_traj.keys():
            cur_traj[key] = np.array(cur_traj[key])
        
        # check undone trajectories
        #if cur_traj['terminals'][-1] == False or bool(cur_traj['success'][-1]) == False:
        if cur_traj['terminals'][-1] == False:
            print(f"Warning: {env_name}-{subtask_idx}-{traj_ind}: Undone trajectory!")
            print("Trajectory length: ", cur_traj['actions'].shape[0])
            print("Done: ", cur_traj['terminals'][-1])
            #print("Success: ", cur_traj['success'][-1])
        
        # add to the pool
        input_trajectories.append(cur_traj)

    # sample prompt trajectories
    prompt_trajectories = random.sample(input_trajectories, prompt_traj_per_subtask)
    
    # set up save folder
    folder_name = f'ML1-{env_name}'
    folder_path = os.path.join(data_path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # save input trajectories 
    input_traj_file_name = f'ML1-{env_name}-{subtask_idx}-expert.pkl'
    input_traj_file_path = os.path.join(folder_path, input_traj_file_name)
    with open(input_traj_file_path, 'wb') as f:
        pickle.dump(input_trajectories, f)
        print("Total input trajectories: ", len(input_trajectories))
        print("======> Saved input trajectories to ", input_traj_file_path)
    
    # save prompt trajectories
    prompt_traj_file_name = f'ML1-{env_name}-{subtask_idx}-prompt-expert.pkl'
    prompt_traj_file_path = os.path.join(folder_path, prompt_traj_file_name)
    with open(prompt_traj_file_path, 'wb') as f:
        pickle.dump(prompt_trajectories, f)
        print("Total prompt trajectories: ", len(prompt_trajectories))
        print("======> Saved input trajectories to ", prompt_traj_file_path)

def save_subtask(subtask, env_name, subtask_idx):
    # set up save folder
    folder_name = f'ML1-{env_name}'
    folder_path = os.path.join(task_config_path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    task_file_name = f'config-ML1-{env_name}-task{subtask_idx}.pkl'
    task_file_path = os.path.join(folder_path, task_file_name)
    with open(task_file_path, 'wb') as f:
        pickle.dump(subtask, f)
        print("======> Saved task config to ", task_file_path)


def generate_ml1(env_name):
    # seed everything except environments
    seed_other(seed=1)

    input_traj_per_subtask = 100
    prompt_traj_per_subtask = 5

    # Construct and seed the benchmark, sampling tasks
    # Note that this seed should be consistent with the one used in training and test
    ml1 = metaworld.ML1(env_name, seed=1) # construct the benchmark, sampling tasks
    # create env
    env = ml1.train_classes[env_name]()  
    # create expert policy
    policy = policies[env_name]() 

    # loop over 50 sub-envs
    for subtask_idx in range(50):
        # Associate env with the subtask
        subtask = ml1.train_tasks[subtask_idx]
        env.set_task(subtask)
        # set goal as observable
        env._partially_observable = False
        # generate input trajectories and prompt trajectories
        generate_one_subtask(env, policy, env_name, subtask_idx, input_traj_per_subtask, prompt_traj_per_subtask)
        # save subtask
        save_subtask(subtask, env_name, subtask_idx)
    
    print("Done!")

def test_scripted_policy(env_name):
    # seed everything except environments
    seed_other(seed=1)

    ml1 = metaworld.ML1(env_name, seed=1) # construct the benchmark, sampling tasks
    # create env
    env = ml1.train_classes[env_name]() 

    # create expert policy
    policy = policies[env_name]() 

    # Associate env with the first subtask
    subtask = ml1.train_tasks[0]
    env.set_task(subtask)
    # set goal as observable
    env._partially_observable = False

    obs = env.reset()
    for step in range(env.max_path_length+1): 
        # no noise added to the action
        action = policy.get_action(obs)
        obs, reward, done, info = env.step(action)
        success = info['success']
        print(success)
        # modify done if succeed
        # done when first succeed
        # if bool(success):
        #     done = True
        
        if done:
            break

    assert done == True
    print('Total timesteps: %d'%(step+1))      

if __name__ == '__main__':
    # generate_ml1('pick-place-v2')
    # generate_ml1('push-v2')
    generate_ml1('reach-v2')

    #test_scripted_policy('pick-place-v2')
    #test_scripted_policy('push-v2')

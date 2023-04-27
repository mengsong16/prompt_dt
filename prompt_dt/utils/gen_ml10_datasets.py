import numpy as np
import gym
import random
import json, pickle, random, os, torch
import copy
from prompt_dt.prompt_utils import get_env_goal
from prompt_dt.utils.other import seed_other
from prompt_dt.utils.path import *

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
    print(f"================= ML10-{env_name}-{subtask_idx}-expert ==================")

    input_trajectories = []

    for traj_ind in range(input_traj_per_subtask): 
        # note that goal_distance, grasp_success is not a general component in info, but 'success'[float] is 
        cur_traj = { 'observations': [], 'actions': [], 'rewards': [], 'terminals': [], 'success': [] }

        env.reset()
        env.reset_model()
        obs = env.reset()
        cur_traj['observations'].append(obs)
        
        for step in range(env.max_path_length): # max path length = 500 from https://github.com/Farama-Foundation/Metaworld/blob/master/metaworld/envs/mujoco/mujoco_env.py
            # no noise added to the action
            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)
            success = info['success']
            # modify done if succeed
            if bool(success):
                done = True

            cur_traj['observations'].append(obs)
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
        
        # check unsuccessful trajectories
        if cur_traj['terminals'][-1] == False or bool(cur_traj['success'][-1]) == False:
            print(f"{env_name}-{subtask_idx}-{traj_ind}: Unsuccessful trajectory!")
            print("Trajectory length: ", cur_traj['actions'].shape[0])
            print("Done: ", cur_traj['terminals'][-1])
            print("Success: ", cur_traj['success'][-1])
        
        # add to the pool
        input_trajectories.append(cur_traj)

    # sample prompt trajectories
    prompt_trajectories = random.sample(input_trajectories, prompt_traj_per_subtask)
    prompt_trajectories = copy.deepcopy(prompt_trajectories)
    
    # set up save folder
    folder_name = f'ML10-{env_name}'
    folder_path = os.path.join(data_path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # save input trajectories 
    input_traj_file_name = f'ML10-{env_name}-{subtask_idx}-expert.pkl'
    input_traj_file_path = os.path.join(folder_path, input_traj_file_name)
    with open(input_traj_file_path, 'wb') as f:
        pickle.dump(input_trajectories, f)
        print("Total input trajectories: ", len(input_trajectories))
        print("======> Saved input trajectories to ", input_traj_file_path)
    
    # save prompt trajectories
    prompt_traj_file_name = f'ML10-{env_name}-{subtask_idx}-prompt-expert.pkl'
    prompt_traj_file_path = os.path.join(folder_path, prompt_traj_file_name)
    with open(prompt_traj_file_path, 'wb') as f:
        pickle.dump(prompt_trajectories, f)
        print("Total prompt trajectories: ", len(prompt_trajectories))
        print("======> Saved input trajectories to ", prompt_traj_file_path)
    
def generate_ml10(seed=1):
    # seed everything except environments
    seed_other(seed=seed)

    input_traj_per_subtask = 100
    prompt_traj_per_subtask = 5

    # Construct and seed the benchmark, sampling tasks
    ml10 = metaworld.ML10(seed=seed) 

    # 10 train envs
    train_envs = []
    train_env_names = []
    train_tasks = [] # a list of list
    for name, env_cls in ml10.train_classes.items():
        env = env_cls()

        sub_tasks = [task for task in ml10.train_tasks
                    if task.env_name == name]
        train_tasks.append(sub_tasks)
        train_envs.append(env)
        train_env_names.append(name)
    
    assert len(train_envs) == len(train_env_names) == len(train_tasks), "ML10: train envs, train_env_names, train_tasks should have the same length"
    
    # loop over base env
    for i, env_name in enumerate(train_env_names):
        
        env = train_envs[i]
        subtasks = train_tasks[i]
        policy = policies[env_name]() # get expert policy

        # loop over sub env
        for subtask_idx in range(len(subtasks)):
            # Associate env with the subtask
            env.set_task(subtasks[subtask_idx])
            # set goal as observable
            env._partially_observable = False
            env._freeze_rand_vec = False
            env._set_task_called = True
            # generate input trajectories and prompt trajectories
            generate_one_subtask(env, policy, env_name, subtask_idx, input_traj_per_subtask, prompt_traj_per_subtask)


### ---------- test scripted policy -------------
def trajectory_summary(env, policy, act_noise_pct, render=False, end_on_success=True):
    """Tests whether a given policy solves an environment
    Args:
        env (metaworld.envs.MujocoEnv): Environment to test
        policy (metaworld.policies.policies.Policy): Policy that's supposed to
            succeed in env
        act_noise_pct (np.ndarray): Decimal value(s) indicating std deviation of
            the noise as a % of action space
        render (bool): Whether to render the env in a GUI
        end_on_success (bool): Whether to stop stepping after first success
    Returns:
        (bool, np.ndarray, np.ndarray, int): Success flag, Rewards, Returns,
            Index of first success
    """
    success = False
    first_success = 0
    rewards = []

    for t, (r, done, info) in enumerate(trajectory_generator(env, policy, act_noise_pct, render)):
        rewards.append(r)
        assert not env.isV2 or set(info.keys()) == {
            'success',
            'near_object',
            'grasp_success',
            'grasp_reward',
            'in_place_reward',
            'obj_to_target',
            'unscaled_reward'
        }
        success |= bool(info['success'])
        if not success:
            first_success = t
        if (success or done) and end_on_success:
            break

    rewards = np.array(rewards)
    returns = np.cumsum(rewards)

    return success, rewards, returns, first_success


def trajectory_generator(env, policy, act_noise_pct, render=False):
    """Tests whether a given policy solves an environment
    Args:
        env (metaworld.envs.MujocoEnv): Environment to test
        policy (metaworld.policies.policies.Policy): Policy that's supposed to
            succeed in env
        act_noise_pct (np.ndarray): Decimal value(s) indicating std deviation of
            the noise as a % of action space
        render (bool): Whether to render the env in a GUI
    Yields:
        (float, bool, dict): Reward, Done flag, Info dictionary
    """
    action_space_ptp = env.action_space.high - env.action_space.low

    env.reset()
    env.reset_model()
    o = env.reset()
    assert o.shape == env.observation_space.shape
    assert env.observation_space.contains(o), obs_space_error_text(env, o)

    for _ in range(env.max_path_length):
        a = policy.get_action(o)
        a = np.random.normal(a, act_noise_pct * action_space_ptp)

        o, r, done, info = env.step(a)
        assert env.observation_space.contains(o), obs_space_error_text(env, o)
        if render:
            env.render()

        yield r, done, info


def obs_space_error_text(env, obs):
    return "Obs Out of Bounds\n\tlow: {}, \n\tobs: {}, \n\thigh: {}".format(
        env.observation_space.low[[0, 1, 2, -3, -2, -1]],
        obs[[0, 1, 2, -3, -2, -1]],
        env.observation_space.high[[0, 1, 2, -3, -2, -1]]
    )


def test_scripted_policy():
    ml10 = metaworld.ML10(seed=1) 
    

    env = None
    for name, env_cls in ml10.train_classes.items():
        if name == 'reach-v2':
            env = env_cls()
            break
    if env is None:
        print("Env is not correctly created!")
        exit()
    
    sub_tasks = [task for task in ml10.train_tasks
                        if task.env_name == 'reach-v2']

    env.set_task(sub_tasks[0])
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True

    policy = SawyerReachV2Policy()
    summary = trajectory_summary(env, policy, act_noise_pct=0, render=False)
    print(summary[0])

if __name__ == '__main__':
    generate_ml10()
    #test_scripted_policy()

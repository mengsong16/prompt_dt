import numpy as np
import gym
import json, pickle, random, os, torch
from collections import namedtuple
from prompt_dt.prompt_utils import load_train_test_env_name_list
from prompt_dt.utils.path import *
from prompt_dt.prompt_utils import get_env_goal

# for mujoco tasks
from mujoco_control_envs.mujoco_control_envs import HalfCheetahDirEnv, HalfCheetahVelEnv, AntDirEnv, WalkerRandParamsWrappedEnv
# for metaworld
import metaworld
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)


# create a single environment with reward scale, maximum episode length, return target
def create_env(env_name, config_save_path):
    if 'cheetah_dir' in env_name:
        # include_goal = False: do not include goal in observations
        if '0' in env_name:  # direction 1
            # tasks = [{'direction': 1}]
            env = HalfCheetahDirEnv([{'direction': 1}], include_goal = False)
        elif '1' in env_name: # direction -1
            env = HalfCheetahDirEnv([{'direction': -1}], include_goal = False)
        
    elif 'cheetah_vel' in env_name:
        task_idx = int(env_name.split('-')[-1])
        task_paths = f"{config_save_path}/cheetah_vel/config_cheetah_vel_task{task_idx}.pkl"
        tasks = []
        with open(task_paths.format(task_idx), 'rb') as f:
            task_info = pickle.load(f)
            assert len(task_info) == 1, f'Unexpected task info: {task_info}'
            tasks.append(task_info[0])
        # print(tasks[0])  # env._goal = 0.075
        # include_goal = False: do not include goal in observations
        env = HalfCheetahVelEnv(tasks, include_goal = False)
        
    elif 'ant_dir' in env_name:
        task_idx = int(env_name.split('-')[-1])
        task_paths = f"{config_save_path}/ant_dir/config_ant_dir_task{task_idx}.pkl"
        tasks = []
        with open(task_paths.format(task_idx), 'rb') as f:
            task_info = pickle.load(f)
            assert len(task_info) == 1, f'Unexpected task info: {task_info}'
            tasks.append(task_info[0])
        # print(tasks[0]) # {'goal': 1.2033605945679715} 
        # env._goal = 1.2033605945679715
        # include_goal = False: do not include goal in observations
        env = AntDirEnv(tasks, len(tasks), include_goal = False)
    elif 'walker_params' in env_name:
        env = WalkerRandParamsWrappedEnv(n_tasks=50)
        env.set_task_idx(0)
    elif 'ML1-' in env_name: # metaworld ML1
        task_name = '-'.join(env_name.split('-')[1:-1])
        # construct the benchmark
        # note that this seed must be 1 which is used to generate the dataset
        ml1 = metaworld.ML1(task_name, seed=1) 
        # construct the environment
        env = ml1.train_classes[task_name]()  
        task_idx = int(env_name.split('-')[-1])
        # task = ml1.train_tasks[task_idx]
        # load task
        task_path = os.path.join(config_save_path, f'ML1-{task_name}', f'config-ML1-{task_name}-task{task_idx}.pkl')
        with open(task_path, 'rb') as f:
            task = pickle.load(f)

        # set task
        env.set_task(task)  
        # print(env.goal) [0.1 0.8 0.2]
    elif 'ML10-' in env_name: # metaworld ML10
        task_name = '-'.join(env_name.split('-')[1:-1])
        task_idx = int(env_name.split('-')[-1])

        # construct the benchmark
        # note that this seed must be 1 which is used to generate the dataset
        ml10 = metaworld.ML10(seed=1) 
        # construct the environment
        env = ml10.train_classes[task_name]()
        # load task
        task_path = os.path.join(config_save_path, f'ML10-{task_name}', f'config-ML10-{task_name}-task{task_idx}.pkl')
        
        with open(task_path, 'rb') as f:
            task = pickle.load(f)

        # set task
        env.set_task(task)

        # env._partially_observable = False
        # env._freeze_rand_vec = False
        # env._set_task_called = True    
    else:
        raise NotImplementedError
    
    return env

def create_env_list(env_name_list, config_save_path):
    env_list = []
    
    for env_name in env_name_list:
        env = create_env(env_name=env_name, config_save_path=config_save_path)
        env_list.append(env)
    return env_list

def test_envs(render):
    # 'ant_dir-0', 'cheetah_vel-0', 'cheetah_dir-0', 'ML1-pick-place-v2-0', "ML10-pick-place-v2-0"
    env_name = 'ML10-pick-place-v2-0'
    env = create_env(env_name=env_name, config_save_path=task_config_path)
    if "ML1" in env_name or "ML10" in env_name:
        max_ep_len = env.max_path_length #500
    else:
        max_ep_len = env._max_episode_steps #200

    # suppress scientific notation
    np.set_printoptions(suppress=True)
    # set print precision
    np.set_printoptions(precision=2)


    for episode in range(100):
        print("--------------------------------------")
        print('Episode: {}'.format(episode))
        obs = env.reset()
        print("state s0: %s"%obs)

        for i in range(max_ep_len): 
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            #print('action: {}   state: {}  reward: {}'.format(action, obs, reward))
            if i == 0:
                print("state s1: %s"%obs)

            if render:
                env.render()

            if done: 
                print("last state: %s"%(obs))
                break
                    
        print('Episode {} finished after {} timesteps.'.format(episode,i+1))
    
    print('-----------------------------')
    print("Observation space: ", env.observation_space)
    print("Action space: ", env.action_space)
    print("Goal: ", get_env_goal(env_name, env))
    print('-----------------------------')

def print_type_shape(traj):
    for key in traj.keys():
        print("-"*80)
        print(key)
        print(traj[key].dtype)
        print(traj[key].shape)

def check_expert_trajectory():
    # env_name = 'cheetah_vel-0'
    # base_env = 'cheetah_vel'

    # env_name = 'ML1-pick-place-v2-0'
    # base_env = 'ML1-pick-place-v2'

    env_name = 'ML10-reach-v2-2'
    base_env = 'ML10-reach-v2'

    env = create_env(env_name=env_name, config_save_path=task_config_path)
    if "ML1" in env_name or "ML10" in env_name:
        max_ep_len = env.max_path_length #500
    else:
        max_ep_len = env._max_episode_steps #200

    # load expert trajectory
    dataset_path = data_path+f'/{base_env}/{env_name}-expert.pkl'
    with open(dataset_path, 'rb') as f:
        expert_trajectories = pickle.load(f)

    expert_traj = expert_trajectories[0]

    print("================= expert trajectory ==================")
    print_type_shape(expert_traj)

    if "ML1" in env_name or "ML10" in env_name:
        env_traj = { 'observations': [], 'actions': [], 'rewards': [], 'terminals': [], 'success': []}
    else:
        env_traj = { 'observations': [], 'actions': [], 'rewards': [], 'terminals': []}

    obs = env.reset()
    env_traj['observations'].append(obs)

    #for i in range(max_ep_len): 
    for i in range(len(expert_traj["actions"])):
        action = expert_traj["actions"][i]
        env_traj['actions'].append(action)

        obs, reward, done, info = env.step(action)
        env_traj['observations'].append(obs)
        env_traj['rewards'].append(reward)
        env_traj['terminals'].append(done)
        
        if 'success' in env_traj.keys():
            env_traj['success'].append(info['success'])

        if done: 
            break
                    
    print('Total timesteps: %d'%(i+1))

    # convert from list to numpy array for each value
    for key in env_traj.keys():
        env_traj[key] = np.array(env_traj[key])
    
    print("================= environment trajectory ==================")
    print_type_shape(env_traj)


    print("================= Compare two trajectories ==================")
    print("================= Compare s0 ==================")
    print(env_traj["observations"][0])
    print('-'*80)
    print(expert_traj["observations"][0])
    #exit()
    print("================= Compare sT ==================")
    print(env_traj["observations"][-1])
    print('-'*80)
    print(env_traj["observations"][-2])
    print('-'*80)
    print(expert_traj["observations"][-1])
    print("================= Compare r0 ==================")
    print(env_traj["rewards"][0])
    print('-'*80)
    print(env_traj["rewards"][0])
    print("================= Compare rT ==================")
    print(env_traj["rewards"][-1])
    print('-'*80)
    print(env_traj["rewards"][-1])
    print("================= Compare d0 ==================")
    print(env_traj["terminals"][0])
    print('-'*80)
    print(env_traj["terminals"][0])
    print("================= Compare dT ==================")
    print(env_traj["terminals"][-1])
    print('-'*80)
    print(env_traj["terminals"][-1])
    print("================= Compare a0 ==================")
    print(env_traj["actions"][0])
    print('-'*80)
    print(env_traj["actions"][0])
    print("================= Compare aT ==================")
    print(env_traj["actions"][-1])
    print('-'*80)
    print(env_traj["actions"][-1])

    if 'success' in env_traj.keys():
        print("================= Compare info0 ==================")
        print(env_traj["success"][0])
        print('-'*80)
        print(env_traj["success"][0])
        print("================= Compare infoT ==================")
        print(env_traj["success"][-1])
        print('-'*80)
        print(env_traj["success"][-1])
    

def test_walker():
    env = WalkerRandParamsWrappedEnv(n_tasks=50)
    #print(env.tasks[0])
    
    for task_id in range(50):
        # env.reset()
        env.set_task_idx(task_id) # already call env.reset here
        #env.reset_task(task_id)

        # print goal parameters
        # print("="*80)
        # print(env._task)
        # print("-"*80)
        # print(env.cur_params)
        # print("-"*80)
        # print(env.get_goal_vector()) # 65
        # print("="*80)

        # print max episode steps
        max_episode_steps = env._max_episode_steps
        print('-----------------------------')
        print("Max episode steps: ", max_episode_steps)
        print("Observation space: ", env.observation_space)
        print("Action space: ", env.action_space)
        #print("Goal: ", get_env_goal(env_name, env))
        print('-----------------------------')
        #print("="*80)
        #print(env.unwrapped._max_episode_steps)
        #print(env.spec._max_episode_steps)

        
        # test in current env for N steps
        for step in range(max_episode_steps):
            env.render()
            # take a random action
            env.step(env.action_space.sample())
            #print(step)  

def test_ml10():
    ml10 = metaworld.ML10(seed=1) # Construct the benchmark, sampling tasks

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
    print("="*80)
    
    for i, env_name in enumerate(train_env_names):
        
        env = train_envs[i]
        subtasks = train_tasks[i]
        # randomly sample a subtask from the subtask list
        subtask = random.choice(subtasks)
        # Associate env with the task
        env.set_task(subtask)

        print('-----------------------------')
        print("Testing ", env_name)
        #print("Max episode steps: ", env._max_episode_steps)
        print("Max episode steps: ", env.max_path_length)
        print("Observation space: ", env.observation_space)
        print("Action space: ", env.action_space)
        #print("Goal: ", get_env_goal(env_name, env))
        print('-----------------------------')

        # test in current env for N steps
        #max_episode_steps
        obs = env.reset()
        for step in range(env.max_path_length):
            # take a random action
            obs, reward, done, info = env.step(env.action_space.sample())
            
            # if step == 1:
            #     print(obs)

            # assert goal is hidden
            assert (obs[-3:] == np.zeros(3)).all() 
            
            env.render()

    # 5 test envs
    # testing_envs = []
    # for name, env_cls in ml10.test_classes.items():
    #     env = env_cls()
    #     print(name)
    #     # task = random.choice([task for task in ml10.test_tasks
    #     #                         if task.env_name == name])
    #     # env.set_task(task)
    #     testing_envs.append(env)

def test_ml10_name():
    ml10 = metaworld.ML10(seed=1) # Construct the benchmark, sampling tasks

    # 10 train envs
    train_envs = []
    train_env_names = []
    train_task_names = [] # a list of list
    for name, env_cls in ml10.train_classes.items():
        env = env_cls()

        sub_task_names = []
        for task in ml10.train_tasks:
            # task.env_name, task.data
            if task.env_name == name:
                sub_task_names.append(task.env_name)

        train_task_names.append(sub_task_names)
        train_envs.append(env)
        train_env_names.append(name)
    
    assert len(train_envs) == len(train_env_names) == len(train_task_names), "ML10: train envs, train_env_names, train_tasks should have the same length"
    
    print("="*80)
    print(train_env_names[0])
    print(train_task_names[0])

def visualize_expert_trajectory():
    # env_name = 'cheetah_vel-0'
    # base_env = 'cheetah_vel'

    # env_name = 'ML1-pick-place-v2-1'
    # base_env = 'ML1-pick-place-v2'

    # env_name = 'ML1-reach-v2-1'
    # base_env = 'ML1-reach-v2'

    # env_name = 'ML10-pick-place-v2-36'
    # base_env = 'ML10-pick-place-v2'

    # env_name = 'ML10-reach-v2-30'
    # base_env = 'ML10-reach-v2'

    env_name = 'ML10-drawer-close-v2-1'
    base_env = 'ML10-drawer-close-v2'

    env = create_env(env_name=env_name, config_save_path=task_config_path)
    if "ML1" in env_name or "ML10" in env_name:
        max_ep_len = env.max_path_length #500
    else:
        max_ep_len = env._max_episode_steps #200

    # load expert trajectory
    dataset_path = data_path+f'/{base_env}/{env_name}-expert.pkl'
    with open(dataset_path, 'rb') as f:
        expert_trajectories = pickle.load(f)

    expert_traj = expert_trajectories[0]

    print("================= expert trajectory ==================")
    
    while True:
        # env.reset()
        # env.reset_model()
        obs = env.reset()
        #for i in range(max_ep_len): 
        for i in range(len(expert_traj["actions"])): 
            action = expert_traj["actions"][i]
            #print(action)
            obs, reward, done, info = env.step(action)
            if "ML1" in env_name or "ML10" in env_name:
                success = info['success']
                print(success)
            env.render()
            
            if done: 
                break
                        
        print('Total timesteps: %d'%(i+1))
        #break

def check_max_ep_len():
    base_env = 'ML1-pick-place-v2' #100
    #base_env = 'ML1-reach-v2' #500
    #base_env = 'ML1-push-v2' #500
    #base_env = 'ML1-door-open-v2' #500
    #base_env = 'ML1-drawer-close-v2' #500
    #base_env = 'ML1-button-press-topdown-v2' #500
    #base_env = 'ML1-peg-insert-side-v2' #500
    #base_env = 'ML1-window-open-v2' #500
    #base_env = 'ML1-sweep-v2' #500
    #base_env = 'ML1-basketball-v2' #500

    #base_env = 'cheetah_dir' #200
    #base_env = 'cheetah_vel' #200
    #base_env = 'ant_dir' #200
    #base_env = 'walker_params' #200


    for i in range(1):
        env_name = f'{base_env}-{i}'
        env = create_env(env_name=env_name, config_save_path=task_config_path)
        # not always 500
        if 'ML1' in env_name or 'ML10' in env_name:
            print(env.max_path_length)
        else:
            print(env._max_episode_steps)
    


if __name__ == '__main__':
    #test_envs(render=True)
    #test_walker()
    #test_ml10()
    #check_expert_trajectory()
    #test_ml10_name()
    visualize_expert_trajectory()
    #check_max_ep_len()


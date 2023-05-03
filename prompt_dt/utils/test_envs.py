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
    elif 'walker_param' in env_name:
        #env = WalkerRandParamsWrappedEnv(n_tasks=50)
        #env.set_task_idx(0)
        task_idx = int(env_name.split('-')[-1])
        task_paths = f"{config_save_path}/walker_param/config_walker_param_task{task_idx}.pkl"
        tasks = []
        with open(task_paths.format(task_idx), 'rb') as f:
            task_info = pickle.load(f)
            assert len(task_info) == 1, f'Unexpected task info: {task_info}'
            tasks.append(task_info[0])
            #print(task_info)

        # include_goal = False: do not include goal in observations
        env = WalkerRandParamsWrappedEnv(tasks, include_goal = False)
    elif 'ML1-' in env_name: # metaworld ML1
        task_name = '-'.join(env_name.split('-')[1:-1])
        # construct the benchmark
        # note that this seed must be 1 which is used to generate the dataset
        ml1 = metaworld.ML1(task_name, seed=1) 
        # construct the environment
        env = ml1.train_classes[task_name]()  
        task_idx = int(env_name.split('-')[-1])

        # print(env_name) # ML1-pick-place-v2-0
        # print(task_name) # pick-place-v2
        # print(task_idx) # 0

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
    #env_name = 'ML10-pick-place-v2-0'
    env_name = 'walker_param-1'
    env = create_env(env_name=env_name, config_save_path=task_config_path)
    if "ML1" in env_name or "ML10" in env_name:
        max_ep_len = env.max_path_length+1 #500
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

            # assert goal is hidden
            if "ML1" in env_name or "ML10" in env_name:
                assert (obs[-3:] == np.zeros(3)).all()

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

def check_trajectory(quality, prompt):
    # env_name = 'cheetah_dir-0'
    # base_env = 'cheetah_dir'

    # env_name = 'cheetah_vel-0'
    # base_env = 'cheetah_vel'

    # env_name = 'ant_dir-0'
    # base_env = 'ant_dir'

    env_name = 'walker_param-10'
    base_env = 'walker_param'

    # env_name = 'ML1-pick-place-v2-0'
    # base_env = 'ML1-pick-place-v2'

    # env_name = 'ML1-reach-v2-0'
    # base_env = 'ML1-reach-v2'

    # env_name = 'ML1-push-v2-0'
    # base_env = 'ML1-push-v2'

    # env_name = 'ML10-reach-v2-2'
    # base_env = 'ML10-reach-v2'

    # env_name = 'ML10-pick-place-v2-0'
    # base_env = 'ML10-pick-place-v2'

    # env_name = 'ML10-peg-insert-side-v2-45'
    # base_env = 'ML10-peg-insert-side-v2'

    env = create_env(env_name=env_name, config_save_path=task_config_path)
    if "ML1" in env_name or "ML10" in env_name:
        max_ep_len = env.max_path_length+1 #500
    else:
        max_ep_len = env._max_episode_steps #200

    # load expert trajectory
    if prompt:
        dataset_path = data_path+f'/{base_env}/{env_name}-prompt-{quality}.pkl'
    else:
        dataset_path = data_path+f'/{base_env}/{env_name}-{quality}.pkl'
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

    if "ML1" in env_name or "ML10" in env_name:
        loop_steps = max_ep_len
    else:
        loop_steps = len(expert_traj["actions"])
    
    for i in range(loop_steps): 
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
    # assert goal is hidden
    if "ML1" in env_name or "ML10" in env_name:
        assert (expert_traj["observations"][0][-3:] == np.zeros(3)).all()
    
    print("================= Compare sT ==================")
    print(env_traj["observations"][-1])
    print('-'*80)
    print(env_traj["observations"][-2])
    print('-'*80)
    print(expert_traj["observations"][-1])
    # assert goal is hidden
    if "ML1" in env_name or "ML10" in env_name:
        assert (expert_traj["observations"][-1][-3:] == np.zeros(3)).all()
    
    exit()

    print("================= Compare r0 ==================")
    print(env_traj["rewards"][0])
    print('-'*80)
    print(expert_traj["rewards"][0])
    print("================= Compare rT ==================")
    print(env_traj["rewards"][-1])
    print('-'*80)
    print(expert_traj["rewards"][-1])
    print("================= Compare d0 ==================")
    print(env_traj["terminals"][0])
    print('-'*80)
    print(expert_traj["terminals"][0])
    print("================= Compare dT ==================")
    print(env_traj["terminals"][-1])
    print('-'*80)
    print(expert_traj["terminals"][-1])
    print("================= Compare a0 ==================")
    print(env_traj["actions"][0])
    print('-'*80)
    print(expert_traj["actions"][0])
    print("================= Compare aT ==================")
    print(env_traj["actions"][-1])
    print('-'*80)
    print(expert_traj["actions"][-1])

    if 'success' in env_traj.keys():
        print("================= Compare info0 ==================")
        print(env_traj["success"][0])
        print('-'*80)
        print(expert_traj["success"][0])
        print("================= Compare infoT ==================")
        print(env_traj["success"][-1])
        print('-'*80)
        print(expert_traj["success"][-1])
    

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
        # print('-----------------------------')
        # print("Max episode steps: ", max_episode_steps)
        # print("Observation space: ", env.observation_space)
        # print("Action space: ", env.action_space)
        # #print("Goal: ", get_env_goal(env_name, env))
        print('-----------------------------')
        #print("="*80)
        #print(env.unwrapped._max_episode_steps)
        #print(env.spec._max_episode_steps)

        
        # test in current env for N steps
        ret = 0
        for step in range(max_episode_steps):
            #env.render()
            # take a random action
            obs, reward, done, info = env.step(env.action_space.sample())
            ret += reward
            #print(step)  
        print(f'Task: {task_id} Return: {ret}')

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
        print("Max episode steps: ", env.max_path_length+1)
        print("Observation space: ", env.observation_space)
        print("Action space: ", env.action_space)
        #print("Goal: ", get_env_goal(env_name, env))
        print('-----------------------------')

        # test in current env for N steps
        #max_episode_steps
        obs = env.reset()
        for step in range(env.max_path_length+1):
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

def compare_macaw_different_quality(base_env, task_num):
    quality_groups = ['random', 'medium', 'expert']

    print(f"========================= {base_env} ============================")

    for quality in quality_groups:
        #quality = "expert"
        ret_list = []
        for task_id in list(range(task_num)):
            # create current sub env
            env = create_env(env_name=f'{base_env}-{task_id}', config_save_path=task_config_path)
            if "ML1" in base_env or "ML10" in base_env:
                max_ep_len = env.max_path_length+1 #500
            else:
                max_ep_len = env._max_episode_steps #200
            # load demonstration trajectories from current sub env
            data_folder_path = data_path
            #data_folder_path = "/home/meng/prompt-dt/macaw_data"
            dataset_path = data_folder_path + f'/{base_env}/{base_env}-{task_id}-{quality}.pkl'
            with open(dataset_path, 'rb') as f:
                demon_trajectories = pickle.load(f)
            # only use the first trajectory
            #demon_traj = demon_trajectories[0]
            demon_traj = demon_trajectories[900]

            # print(demon_traj["observations"].shape)
            # print(demon_traj["actions"].shape)
            # print(demon_traj["rewards"].shape)
            # print(demon_traj["terminals"].shape)
            # exit()
             
            if "ML1" in base_env or "ML10" in base_env:
                loop_steps = max_ep_len
            else:
                loop_steps = len(demon_traj["actions"])

            obs = env.reset()
            cur_ret = 0
            for i in range(loop_steps): 
                action = demon_traj["actions"][i]
                # print(action)
                # exit()
                obs, reward, done, info = env.step(action)
                print("-"*80)
                print(reward)
                print(demon_traj["rewards"][i][0])
                cur_ret += reward
                if "ML1" in base_env or "ML10" in base_env:
                    success = info['success']
                    print(success)
                
                #env.render()
                
                if done: 
                    break
            
            print("-"*80)
            print(cur_ret)
            print(demon_traj["rewards"].sum())
            exit()
            ret_list.append(cur_ret)

            if "ML1" in base_env or "ML10" in base_env:        
                assert done == True

            #print('Total timesteps: %d'%(i+1))
        
        ret_list = np.array(ret_list)

        #print("-"*80)
        print(quality, np.mean(ret_list))
        

def visualize_expert_trajectory(quality="expert"):
    # env_name = 'cheetah_vel-20'
    # base_env = 'cheetah_vel'

    # env_name = 'ant_dir-30' #30
    # base_env = 'ant_dir'

    # env_name = 'ML1-pick-place-v2-0'
    # base_env = 'ML1-pick-place-v2'

    # env_name = 'ML1-reach-v2-1'
    # base_env = 'ML1-reach-v2'

    # env_name = 'ML10-pick-place-v2-0'
    # base_env = 'ML10-pick-place-v2'

    # env_name = 'ML10-reach-v2-30'
    # base_env = 'ML10-reach-v2'

    # env_name = 'ML10-drawer-close-v2-0'
    # base_env = 'ML10-drawer-close-v2'

    # env_name = 'ML10-sweep-v2-0'
    # base_env = 'ML10-sweep-v2'

    # env_name = 'ML10-push-v2-0'
    # base_env = 'ML10-push-v2'

    # env_name = 'ML10-window-open-v2-0'
    # base_env = 'ML10-window-open-v2'

    # env_name = 'ML10-basketball-v2-0'
    # base_env = 'ML10-basketball-v2'

    # env_name = 'ML10-peg-insert-side-v2-0'
    # base_env = 'ML10-peg-insert-side-v2'

    # env_name = 'ML10-door-open-v2-0'
    # base_env = 'ML10-door-open-v2'

    env_name = 'walker_param-10'
    base_env = 'walker_param'

    env = create_env(env_name=env_name, config_save_path=task_config_path)
    if "ML1" in env_name or "ML10" in env_name:
        max_ep_len = env.max_path_length+1 #500
    else:
        max_ep_len = env._max_episode_steps #200

    # load expert trajectory
    dataset_path = data_path+f'/{base_env}/{env_name}-{quality}.pkl'
    with open(dataset_path, 'rb') as f:
        expert_trajectories = pickle.load(f)

    expert_traj = expert_trajectories[0]

    print("================= demonstration trajectory ==================")
    if "ML1" in env_name or "ML10" in env_name:
        loop_steps = max_ep_len
    else:
        loop_steps = len(expert_traj["actions"])
    
    while True:
        obs = env.reset() 
        for i in range(loop_steps): 
            # print(expert_traj["actions"].dtype)
            # print(expert_traj["observations"].dtype)
            # print(expert_traj["rewards"].dtype)
            # print(expert_traj["terminals"].dtype)
            # exit()
            
            action = expert_traj["actions"][i]
            #print(action)
            obs, reward, done, info = env.step(action)
            if "ML1" in env_name or "ML10" in env_name:
                success = info['success']
                print(success)
            
            env.render()
            
            if done: 
                break

        if "ML1" in env_name or "ML10" in env_name:        
            assert done == True

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
    #base_env = 'walker_param' #200


    for i in range(1):
        env_name = f'{base_env}-{i}'
        env = create_env(env_name=env_name, config_save_path=task_config_path)
        # not always 500
        if 'ML1' in env_name or 'ML10' in env_name:
            print(env.max_path_length+1)
        else:
            print(env._max_episode_steps)
    


if __name__ == '__main__':
    #test_envs(render=True)
    #test_walker()
    #test_ml10()
    check_trajectory(quality='expert', prompt=False)
    #test_ml10_name()
    #visualize_expert_trajectory(quality="expert")
    #visualize_expert_trajectory(quality="medium")
    #visualize_expert_trajectory(quality="random")
    #check_max_ep_len()

    #compare_macaw_different_quality(base_env="walker_param", task_num=50)
    #compare_macaw_different_quality(base_env="ant_dir", task_num=50)
    #compare_macaw_different_quality(base_env="cheetah_vel", task_num=40)


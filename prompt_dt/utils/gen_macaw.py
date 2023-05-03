import h5py, pickle, os, random, importlib
from tqdm import tqdm
import numpy as np
import metaworld
from prompt_dt.utils.path import *
import os
import random
from prompt_dt.utils.other import seed_other

macaw_pdt_keymap = {
            'obs': 'observations',
            'actions': 'actions',
            'terminals': 'terminals',
            'rewards': 'rewards'
        }

quality_groups = ['random', 'medium', 'expert']

def gen_macaw_one_dataset(base_env, task_index, hdf5_file_path, input_traj_num, prompt_traj_num, save_quality):
    with h5py.File(hdf5_file_path, 'r') as f:
        # check keys and values
        num_transitions = f['obs'].shape[0]
        for k in macaw_pdt_keymap.keys():
            assert k in f.keys(), f"Key error: {k} not in hdf5 keys" 
            assert f[k].shape[0] == num_transitions, "Error: the number of transitions should be equal for all keys"

        # divide transitions into different quality groups for each key 
        data = {"random": {}, "medium": {}, "expert": {}}

        # get done=True positions
        dones = f['terminals']
        
        # find division locations
        # find the first division location
        quality_group_size = num_transitions // 3
        for i in list(range(quality_group_size-1, 2*quality_group_size)):
            if bool(dones[i]) == True:
                first_cut_loc = i
                break
        
        # find the second division location
        for i in list(range(2*quality_group_size-1, num_transitions)):
            if bool(dones[i]) == True:
                second_cut_loc = i
                break
        
        # find the third division location
        for i in list(range(num_transitions-1, second_cut_loc, -1)):
            if bool(dones[i]) == True:
                third_cut_loc = i
                break
        
        # print("-"*80)
        # print(quality_group_size)
        # print(first_cut_loc)
        # print("-"*80)
        # print(quality_group_size*2)
        # print(second_cut_loc)
        # print("-"*80)
        # print(num_transitions)
        # print(third_cut_loc)
        
        for macaw_key in macaw_pdt_keymap.keys():
            # map to pdt key
            pdt_key = macaw_pdt_keymap[macaw_key]
            # divide equally into three groups based on terminals
            data["random"][pdt_key] = np.array(f[macaw_key][:first_cut_loc+1])
            data["medium"][pdt_key] = np.array(f[macaw_key][first_cut_loc+1:second_cut_loc+1])
            data["expert"][pdt_key] = np.array(f[macaw_key][second_cut_loc+1:third_cut_loc+1])
            #assert data["random"][pdt_key].shape[0] + data["medium"][pdt_key].shape[0] + data["expert"][pdt_key].shape[0] == num_transitions, "Error: transition number is wrong after division"
        
        # ensure that done=True at the end of each quality data
        for quality in quality_groups:
            assert bool(data[quality]['terminals'][-1]) == True
        
        # print("-"*80)
        # print(data["random"]["observations"].shape)
        # print(data["medium"]["observations"].shape)
        # print(data["expert"]["observations"].shape)
        # exit()

        # isolate transitions into trajectories
        trajectories = {"random": [], "medium": [], "expert": []}
        # isolate for each quality group
        for quality in quality_groups:
            #cur_quality_trans_num = data[quality]['terminals'].shape[0]

            # convert done to bool array
            data[quality]['terminals'] = np.array(data[quality]['terminals'], dtype=bool)
            # find done=True locations
            terminal_indices = (data[quality]['terminals']==True).nonzero()[0].tolist() 
            
            # isolate into trajectories
            start_index = 0
            for end_index in terminal_indices:
                # trajectory should contain at least one transition
                if end_index <= start_index:
                    continue

                cur_traj = {}
                # get current trajectory
                for k in data[quality].keys():
                    cur_traj[k] = data[quality][k][start_index:(end_index+1)]
                    # each component is a numpy array
                    cur_traj[k] = np.array(cur_traj[k])

                assert cur_traj['terminals'][-1] == True, "Error: Undone trajectory!"
                trajectories[quality].append(cur_traj)
                # next trajectory
                start_index = end_index + 1
        
        # print out summary
        print("Raw data")
        for quality in quality_groups:
            print(quality, len(trajectories[quality]))
        
        # extract 1000 input trajectories for each quality dataset
        input_trajectories = {}
        # make sure that there are enough number of trajectories
        for quality in quality_groups:
            assert len(trajectories[quality]) >= input_traj_num, "Not enough trajectories"

        input_trajectories["random"] = trajectories["random"][:input_traj_num]

        medium_middle_loc = len(trajectories["medium"]) // 2
        medium_first_half_length = input_traj_num // 2
        medium_start = medium_middle_loc - medium_first_half_length
        medium_end = medium_start + input_traj_num
        input_trajectories["medium"] = trajectories["medium"][medium_start:medium_end]

        input_trajectories["expert"] = trajectories["expert"][-input_traj_num:]

        # extract 5 prompt trajectories for each quality dataset
        assert prompt_traj_num <= input_traj_num, "Prompt trajectories should be less than the input trajectories"
        prompt_trajectories = {}
        for quality in quality_groups:
            #sample_indices = random.sample(np.arange(input_traj_num).tolist(), prompt_traj_num)
            #print(sample_indices)
            prompt_trajectories[quality] = random.sample(input_trajectories[quality], prompt_traj_num)

        # print out summary
        print("-"*80)
        print("Extracted data")
        for quality in quality_groups:
            print(quality, len(input_trajectories[quality]))
            print(quality, len(prompt_trajectories[quality]))
        print("-"*80)

        # dump
        #save_folder = os.path.join(data_path, base_env)
        save_folder = os.path.join('/home/meng/prompt-dt/macaw_data', base_env)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for quality in save_quality:
            # dump input trajectories
            input_traj_filename = f'{base_env}-{task_index}-{quality}.pkl'
            input_traj_path = os.path.join(save_folder, input_traj_filename)
            with open(input_traj_path, 'wb') as g:
                pickle.dump(input_trajectories[quality], g)
                print("======> Saved to ", input_traj_path)

            # dump trajectory trajectories
            prompt_traj_filename = f'{base_env}-{task_index}-prompt-{quality}.pkl'
            prompt_traj_path = os.path.join(save_folder, prompt_traj_filename)
            with open(prompt_traj_path, 'wb') as g:
                pickle.dump(prompt_trajectories[quality], g)
                print("======> Saved to ", prompt_traj_path)
            

def gen_macaw_datasets(base_env, task_num, save_quality, input_traj_num=1000, prompt_traj_num=5):
    
    macaw_dataset_path = "/home/meng/macaw_offline_data"
    macaw_base_env_path = os.path.join(macaw_dataset_path, base_env)

    # seed everything
    seed_other(seed=1)

    #for i in tqdm(range(task_num)):
    for i in range(task_num):
        hdf5_file_name = f'buffers_{base_env}_train_{i}_sub_task_0.hdf5'
        hdf5_file_path = os.path.join(macaw_base_env_path, hdf5_file_name)
        print(f'----------------------------- {base_env} Task: {i} -----------------------------------')
        gen_macaw_one_dataset(base_env, i, hdf5_file_path, input_traj_num, prompt_traj_num, save_quality)

        #break
    
    print("Done!")


def rename_task_config():
    env_folder_name = "walker_param"
    task_path = os.path.join(task_config_path, env_folder_name)
    for i in range(50):
        from_name = f'env_{env_folder_name}_train_task{i}.pkl'
        to_name = f'config_{env_folder_name}_task{i}.pkl'
        os.rename(os.path.join(task_path, from_name), os.path.join(task_path, to_name))
        print(from_name, to_name)
    
    print("Name conversion Done!")

if __name__ == '__main__':
    # rename_task_config()

    gen_macaw_datasets(base_env="walker_param", task_num=50, save_quality=['random', 'medium', 'expert'])
    gen_macaw_datasets(base_env="ant_dir", task_num=50, save_quality=['random', 'medium', 'expert'])
    gen_macaw_datasets(base_env="cheetah_vel", task_num=40, save_quality=['random', 'medium', 'expert'])

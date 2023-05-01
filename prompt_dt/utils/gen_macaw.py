import h5py, pickle, os, random, importlib
from tqdm import tqdm
import numpy as np
import metaworld
from prompt_dt.utils.path import *
import os

def convert_macaw(total_tasks, quality_group_size, output_path, input_path, task_name):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    macaw_pdt_keymap = {
            'obs': 'observations',
            'next_obs': 'next_observations',
            'actions': 'actions',
            'terminals': 'terminals',
            'rewards': 'rewards'
        }

    for i in tqdm(range(total_tasks)):
        with h5py.File(f'./{input_path}/buffers_{task_name}_train_{i}_sub_task_0.hdf5', 'r') as f:
            for quality, idx in zip(('random', 'medium', 'expert'), (0, 1, 2)):
                # CREATE DATASETS
                prompt_indices = None
                num_transitions = None
                data = {}
                prompt_data = {}

                for key in f.keys():
                    # only care about these for PDT
                    if key not in ('obs', 'actions', 'rewards', 'next_obs', 'terminals'):
                        continue

                    if f[key].shape == ():
                        continue
                    
                    if num_transitions == None:
                        num_transitions = f[key].shape[0]

                    if type(prompt_indices) == type(None):
                        prompt_indices = np.random.choice(num_transitions, size=5, replace=False)

                    pdt_key = macaw_pdt_keymap[key]
                    if idx == 0: # first N trajectories
                        data[pdt_key] = np.array(f[key][:quality_group_size])
                    elif idx == 1: # middle N trajectories
                        start_idx = (num_transitions - quality_group_size) // 2
                        data[pdt_key] = np.array(f[key][start_idx:start_idx + quality_group_size])
                    else: # final N trajectories
                        data[pdt_key] = np.array(f[key][-quality_group_size:])

                    prompt_data[pdt_key] = np.array([f[key][p_idx] for p_idx in prompt_indices])

                # ISOLATE TRAJECTORIES
                episodes = []
                curr_episode = {}
                for done_idx in range(len(data['terminals'])):
                    for key in data.keys():
                        if key not in curr_episode.keys():
                            curr_episode[key] = []
                        curr_episode[key].append(data[key][done_idx])

                    if data['terminals'][done_idx] == True:
                        for key in curr_episode.keys():
                            curr_episode[key] = np.array(curr_episode[key])
                        episodes.append(curr_episode)
                        curr_episode = {}

                for key in curr_episode.keys():
                    curr_episode[key] = np.array(curr_episode[key])
                episodes.append(curr_episode) # end of dataset
                episodes = np.array(episodes)

                # convert prompt to array
                prompt_data = np.array([prompt_data])

                with open(f'./{output_path}/{task_name}-{i}-{quality}.pkl', 'wb') as g:
                    pickle.dump(episodes, g)

                with open(f'./{output_path}/{task_name}-{i}-prompt-{quality}.pkl', 'wb') as g:
                    pickle.dump(prompt_data, g)

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
    # CONVERT MACAW DATA TO PDT-USABLE FORMAT
    # task_name = 'cheetah_vel'
    # output_path = f'data/{task_name}'
    # input_path = 'data/cheetah_vel_macaw'
    # convert_macaw(50, 100000, output_path, input_path, task_name)
    rename_task_config()

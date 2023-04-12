import os
import time
import torch
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy as sp
import pickle
import random
from prompt_dt.utils.path import *

class GoalStateDataset(Dataset):
    """Goal states dataset."""

    def __init__(self, base_env, split):
        self.load_dataset(base_env, split)
        
    # dataset_mode: ["expert"] 
    # traj_type: ["input", "prompt"]
    # split: ["train", "test"]
    def load_dataset(self, base_env, split, 
                     traj_type="input", dataset_mode="expert"):
        goal_state_folder = os.path.join(data_path, base_env, "goal_states")
        
        if traj_type == "input":
            traj_type_str = ""
        else:
            traj_type_str = "-prompt"
        
        goal_state_file_name = f'{base_env}{traj_type_str}-{dataset_mode}-{split}-goal_states.pkl'
        load_path = os.path.join(goal_state_folder, goal_state_file_name)
        
        with open(load_path, 'rb') as f:
            self.goal_state_list = pickle.load(f)
        
        print('======> Loaded %d goal states from %s'%(len(self.goal_state_list), load_path))
    

    def __len__(self):
        return len(self.goal_state_list)

    # get one sample
    # sample = {'goal': goal, 'goal_state': goal_state}
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        numpy_sample = self.goal_state_list[idx]

        # [goal_dim]
        goal =  torch.from_numpy(numpy_sample["goal"]).to(dtype=torch.float32)
        # [state_dim]
        goal_state = torch.from_numpy(numpy_sample["goal_state"]).to(dtype=torch.float32) 

        return goal, goal_state
    
def get_train_data_loader(variant):
    dataset = GoalStateDataset(variant['base_env'], split="train")

    data_loader = DataLoader(
        dataset=dataset, 
        batch_size=int(variant['batch_size']), 
        shuffle=True,
        num_workers=0)
    
    return data_loader

def get_test_data_loader(variant):
    dataset = GoalStateDataset(variant['base_env'], split="test")

    data_loader = DataLoader(
        dataset=dataset, 
        batch_size=int(variant['batch_size']), 
        shuffle=True,
        num_workers=0)
    
    return data_loader

if __name__ == "__main__":
    # ["cheetah_dir", "cheetah_vel", "ant_dir", "ML1-pick-place-v2"]
    batch_size = 256
    variant = {"base_env": "ML1-pick-place-v2", "batch_size": batch_size}
    data_loader = get_train_data_loader(variant)
    print("The number of data points: ", len(data_loader.dataset))
    print("Batch size: ", batch_size)
    print("The number of batches: ", len(data_loader))

    for iteration, (goals, goal_states) in enumerate(data_loader):
        print("Iteration: ", iteration)
        print(goals.size()) # [batch_size, goal_dim]
        print(goal_states.size()) # [batch_size, state_dim]
        #print(goals[0,:])
        #print(goal_states[0,:])
        print(goals)
        print("-"*80)
    
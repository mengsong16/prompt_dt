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
            idx = idx.tolist()[0]
        
        # randomly pick one sample from the list
        numpy_sample = random.choice(self.goal_state_list)

        sample = {
            'goal': torch.tensor(numpy_sample["goal"], dtype=torch.float), 
            'goal_state': torch.tensor(numpy_sample["goal_state"], dtype=torch.float) 
        }  
        
        return sample
    
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
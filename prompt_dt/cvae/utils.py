import os
import time
import torch
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
import numpy as np
import scipy as sp
import datetime

def set_up_experiment_name_folder(variant, seed):
    group_name = variant['algorithm_name'] 
    if variant['algorithm_name'] == "cvae" or variant['algorithm_name'] == "vae":
        group_name += ("-"+ variant['prior_distribution'])

    now = datetime.datetime.now()
    experiment_name = "s%d-"%(seed) + now.strftime("%Y%m%d-%H%M%S").lower() 
    experiment_folder = os.path.join(group_name, experiment_name)

    return experiment_name, experiment_folder

def save_model(model, runs_path, experiment_folder):
    experiment_runs_path = os.path.join(runs_path, experiment_folder)
    if not os.path.exists(experiment_runs_path):
        os.makedirs(experiment_runs_path)

    torch.save(model.state_dict(), 
               os.path.join(experiment_runs_path, "best_model.pth"))
    

def plot_llk(test_loss_list, test_epoch_list, 
             figure_path, experiment_folder,
             reverse_loss, image_name):
    experiment_figure_path = os.path.join(figure_path, experiment_folder)
    if not os.path.exists(experiment_figure_path):
        os.makedirs(experiment_figure_path)

    test_loss_list = np.array(test_loss_list)
    test_epoch_list = np.array(test_epoch_list)

    plt.figure(figsize=(30, 10))
    sns.set_style("whitegrid")

    # reverse test elbo loss so that higher is better
    if reverse_loss:
        data = np.concatenate(
            [test_epoch_list[:, sp.newaxis], -test_loss_list[:, sp.newaxis]], axis=1
        )
    else:
        data = np.concatenate(
            [test_epoch_list[:, sp.newaxis], test_loss_list[:, sp.newaxis]], axis=1
        )

    df = pd.DataFrame(data=data, columns=["Training Epoch", "Test ELBO"])
    g = sns.FacetGrid(df, aspect=1.5)
    g.map(plt.scatter, "Training Epoch", "Test ELBO")
    g.map(plt.plot, "Training Epoch", "Test ELBO")

    test_save_path = os.path.join(experiment_figure_path,
                image_name)
    plt.savefig(test_save_path, dpi=300)

    plt.clf()
    plt.close("all")

    print("======> Test results saved to ", test_save_path)

# Anneal beta linearly from 0 to 1 over anneal_len epochs
def anneal_beta(epoch, anneal_num_epochs, num_epochs):
    assert anneal_num_epochs <= num_epochs
    
    # epoch index from 0
    beta = float(epoch) / float(anneal_num_epochs)
    # ensure beta is in [0,1] (both inclusive)
    beta = min(beta, 1.0) 

    return beta
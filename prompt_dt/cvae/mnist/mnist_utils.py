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


from prompt_dt.utils.path import *



def get_train_data_loader(variant):
    # each pixel is a float number in [0,1]
    dataset = MNIST(
        root=data_path, train=True, transform=transforms.ToTensor(),
        download=True)

    data_loader = DataLoader(
        dataset=dataset, batch_size=variant['batch_size'], shuffle=True)

    return data_loader

def get_test_data_loader(variant):
    # each pixel is a float number in [0,1]
    dataset = MNIST(
        root=data_path, train=False, transform=transforms.ToTensor(),
        download=True)
    
    data_loader = DataLoader(
        dataset=dataset, batch_size=variant['batch_size'], shuffle=True)

    return data_loader

# c is a list of conditional variables (e.g. digit index)
# generated_x is a list of generated data (e.g. digit image)
def save_generated_images(c, generated_x, epoch, experiment_folder):
    experiment_figure_path = os.path.join(mnist_figure_path, experiment_folder)
    if not os.path.exists(experiment_figure_path):
        os.makedirs(experiment_figure_path)

    plt.figure()
    plt.figure(figsize=(5, 10))
    for p in range(10):
        plt.subplot(5, 2, p+1)
        plt.text(0, 0, "c={:d}".format(c[p].item()), color='black',
            backgroundcolor='white', fontsize=8)
        plt.imshow(generated_x[p].view(28, 28).cpu().data.numpy())
        plt.axis('off')

    plt.savefig(
        os.path.join(experiment_figure_path,
                    "epoch_{:d}_gen_data.png".format(epoch)), dpi=300)
    plt.clf()
    plt.close('all')

# index to one hot vector
def idx2onehot(idx, n):
    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    return onehot

# zs are tensors in the latent space
# classes are tensors of z's class
def plot_tsne(zs, labels, class_number, experiment_folder):
    experiment_figure_path = os.path.join(mnist_figure_path, experiment_folder)
    if not os.path.exists(experiment_figure_path):
        os.makedirs(experiment_figure_path)

    model_tsne = TSNE(n_components=2, random_state=0)

    z_states = zs.detach().cpu().numpy() # (10000, 50)
    z_embed = model_tsne.fit_transform(z_states) # (10000, 2)
    labels = labels.detach().cpu().numpy() # (10000, )

    fig = plt.figure()
    for ic in range(class_number):
        # plot points belong to each class
        ind_class = np.where(labels == ic)[0] # (n, )
        
        color = plt.cm.Set1(ic)
        plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=class_number, color=color)

    # save tsne image
    tsne_save_path = os.path.join(experiment_figure_path,
                    "test_data_tsne_embedding.png")
    fig.savefig(tsne_save_path, dpi=300)

    plt.clf()
    plt.close("all")

    print("======> Tsne embedding results saved to ", tsne_save_path)


def embed_test_dataset(cvae_net, test_data_loader,
                       class_number, experiment_folder, device):
    """
    This is used to generate a t-sne embedding of the vae
    """

    data = test_data_loader.dataset.data.float()
    labels = test_data_loader.dataset.targets

    data = data.to(device)
    labels = labels.to(device)

    zs = cvae_net.embed_x(data, labels)
    
    plot_tsne(zs, labels, class_number, experiment_folder)

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

from prompt_dt.cvae.mnist.mnist_cvae import MnistCVAE
from prompt_dt.utils.other import seed_other
from prompt_dt.utils.path import *
from prompt_dt.utils.other import parse_config
import datetime

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
import pyro.poutine as poutine

from tqdm import tqdm

def get_train_data_loader(variant):
    dataset = MNIST(
        root=data_path, train=True, transform=transforms.ToTensor(),
        download=True)
    
    data_loader = DataLoader(
        dataset=dataset, batch_size=variant['batch_size'], shuffle=True)

    return data_loader

def get_test_data_loader(variant):
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


def train_one_epoch(train_data_loader, svi, device, logs,
                    epoch, variant, verbose=True):
    num_epochs = int(variant['num_epochs'])
    num_train_samples = len(train_data_loader.dataset)
    bar = tqdm(train_data_loader,
            desc="CVAE Epoch {}".format(epoch).ljust(20), # Left justify string of minimum width 20
        )

    # compute beta for current epoch
    anneal_num_epochs = int(variant['anneal_num_epochs'])
    if variant['anneal_beta']:
        beta = anneal_beta(epoch, anneal_num_epochs, num_epochs)
    else:
        beta = int(variant['beta'])
        
    # Train for one epoch: iterate over the entire dataset once
    epoch_elbo_loss = 0
    epoch_kl_loss = 0
    for iteration, (x, c) in enumerate(bar):
        x, c = x.to(device), c.to(device)
        batch_elbo_loss = svi.step(x, c, beta=beta)
        epoch_elbo_loss += batch_elbo_loss
        # extract kl loss (no gradient)
        batch_kl_loss = svi.evaluate_loss(x, c, beta=1.0e-9)
        epoch_kl_loss += batch_kl_loss

    epoch_elbo_loss /= float(num_train_samples)
    epoch_kl_loss /= float(num_train_samples)
    logs['elbo_loss'].append(epoch_elbo_loss)
    logs['kl_loss'].append(epoch_kl_loss)
    
    # verbose
    if verbose:
        print("Epoch {:02d}/{:02d}, train ELBO loss {:9.4f}, train KL loss {:9.4f}".format(
            epoch, num_epochs, epoch_elbo_loss, epoch_kl_loss))

# evaluate loss over test dataset
def test(test_data_loader, svi, device,
         epoch, variant, verbose=True):
    
    num_epochs = int(variant['num_epochs'])

    test_elbo_loss = 0
    test_kl_loss = 0
    # compute the loss over the entire test set
    for iteration, (x, c) in enumerate(test_data_loader):
        x, c = x.to(device), c.to(device)

        # compute ELBO loss and KL loss
        test_elbo_loss += svi.evaluate_loss(x, c, beta=1.0)
        test_kl_loss += svi.evaluate_loss(x, c, beta=1.0e-9)

    # report test loss
    num_test_samples = len(test_data_loader.dataset)
    test_elbo_loss /= float(num_test_samples)
    test_kl_loss /= float(num_test_samples)

    # verbose
    if verbose:
        print('-'*80)
        print("Epoch {:02d}/{:02d}, test ELBO loss {:9.4f}, test KL loss {:9.4f}".format(
            epoch, num_epochs, test_elbo_loss, test_kl_loss))
        print('-'*80)
    
    return test_elbo_loss, test_kl_loss

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

def plot_llk(test_elbo, test_epoch_list, experiment_folder):
    experiment_figure_path = os.path.join(mnist_figure_path, experiment_folder)
    if not os.path.exists(experiment_figure_path):
        os.makedirs(experiment_figure_path)

    test_elbo = np.array(test_elbo)
    test_epoch_list = np.array(test_epoch_list)

    plt.figure(figsize=(30, 10))
    sns.set_style("whitegrid")

    # reverse test elbo loss so that higher is better
    data = np.concatenate(
        [test_epoch_list[:, sp.newaxis], -test_elbo[:, sp.newaxis]], axis=1
    )

    df = pd.DataFrame(data=data, columns=["Training Epoch", "Test ELBO"])
    g = sns.FacetGrid(df)
    g.map(plt.scatter, "Training Epoch", "Test ELBO")
    g.map(plt.plot, "Training Epoch", "Test ELBO")

    test_save_path = os.path.join(experiment_figure_path,
                "test_elbo.png")
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

def save_model(cvae_net, experiment_folder):
    experiment_runs_path = os.path.join(mnist_runs_path, experiment_folder)
    if not os.path.exists(experiment_runs_path):
        os.makedirs(experiment_runs_path)

    torch.save(cvae_net.state_dict(), 
               os.path.join(experiment_runs_path, "best_model.pth"))
    
def experiment():
    # parse config file
    config_file_path = os.path.join(mnist_path, "mnist_config.yaml")
    variant = parse_config(config_file_path)
    # print out variant
    print("=========================== variant ==================================")
    print(variant)

    # clear pyro param store
    pyro.clear_param_store()

    # seed
    seed = int(variant['seed'])
    seed_other(seed)

    # device
    device = variant['device']

    # train data loader
    train_data_loader = get_train_data_loader(variant)

    # test data loader
    test_data_loader = get_test_data_loader(variant)

    # create cvae net and register its parameters to pyro param store
    cvae_net = MnistCVAE(latent_dim=int(variant['latent_dim']), 
                    hidden_dim=int(variant['hidden_dim']), 
                    num_labels=10, 
                    prior_distribution=variant['prior_distribution'])
    cvae_net.to(device)

    # create optimizer
    optimizer = pyro.optim.Adam({"lr": variant['learning_rate']})

    # create ELBO optimizer
    svi = SVI(cvae_net.model, cvae_net.guide, optimizer, loss=Trace_ELBO())

    # set up experiment name
    group_name = variant['algorithm_name'] 
    if variant['algorithm_name'] == "cvae" or variant['algorithm_name'] == "vae":
        group_name += ("-"+ variant['prior_distribution'])

    now = datetime.datetime.now()
    experiment_name = "s%d-"%(seed) + now.strftime("%Y%m%d-%H%M%S").lower() 
    experiment_folder = os.path.join(group_name, experiment_name)

    # prepare training log
    logs = defaultdict(list)
    logs['elbo_loss'] = []
    logs['kl_loss'] = []
    
    # start training
    num_epochs = int(variant['num_epochs'])
    test_frequency = int(variant["test_frequency"])

    test_elbo = []
    test_epoch_list = []
    best_test_elbo_loss = np.inf
    
    for epoch in range(num_epochs):
        # train for one epoch
        train_one_epoch(train_data_loader, svi, device, logs,
                    epoch, variant)
        
        # evaluate
        if epoch % test_frequency == 0 or epoch == num_epochs - 1:
            # evaluate over test dataset
            test_elbo_loss, _ = test(test_data_loader, svi, device, epoch, variant)
            test_elbo.append(test_elbo_loss)
            test_epoch_list.append(epoch)

            # save best model until now
            if test_elbo_loss < best_test_elbo_loss:
                best_test_elbo_loss = test_elbo_loss
                save_model(cvae_net, experiment_folder)
                print("Better performance achieved: %f, model saved"%(best_test_elbo_loss))

            # generate data and save
            c = torch.arange(0, 10).long().unsqueeze(1).to(device)
            
            generated_x = cvae_net.generate_x(c, deterministic=True)
            save_generated_images(c, generated_x, epoch, experiment_folder)
            

    # plot and save tsne 
    embed_test_dataset(cvae_net, test_data_loader,
                       class_number=10, 
                       experiment_folder=experiment_folder,
                       device=device)

    # plot and save test elbo during the training process
    plot_llk(test_elbo, test_epoch_list, experiment_folder)


if __name__ == '__main__':
    experiment()

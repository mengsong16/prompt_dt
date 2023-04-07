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

from prompt_dt.cvae.mnist.mnist_cvae import ImageCVAE
from prompt_dt.utils.other import seed_other
from prompt_dt.utils.path import *
from prompt_dt.utils.other import parse_config
import datetime

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
import pyro.poutine as poutine

from tqdm import tqdm

from prompt_dt.cvae.mnist.mnist_utils import get_train_data_loader, get_test_data_loader, save_generated_images, embed_test_dataset 
from prompt_dt.cvae.utils import set_up_experiment_name_folder, save_model, plot_llk, anneal_beta


def train_one_epoch(train_data_loader, svi, device, logs,
                    epoch, variant, verbose=True):
    num_epochs = int(variant['num_epochs'])
    num_train_samples = len(train_data_loader.dataset) # len(train_data_loader) is the number of batches
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


def experiment():
    # parse config file
    config_file_path = os.path.join(mnist_path, "mnist_cvae_config.yaml")
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
    cvae_net = ImageCVAE(latent_dim=int(variant['latent_dim']), 
                    hidden_dim=int(variant['hidden_dim']), 
                    num_labels=10, 
                    image_height=28, image_width=28,
                    prior_distribution=variant['prior_distribution'])
    cvae_net.to(device)

    # create optimizer
    optimizer = pyro.optim.Adam({"lr": variant['learning_rate']})

    # create ELBO optimizer
    svi = SVI(cvae_net.model, cvae_net.guide, optimizer, loss=Trace_ELBO())

    # set up experiment name
    experiment_name, experiment_folder = set_up_experiment_name_folder(variant, seed)

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
                save_model(model=cvae_net, 
                           runs_path=mnist_runs_path,
                           experiment_folder=experiment_folder)
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
    plot_llk(test_elbo, test_epoch_list, 
             mnist_figure_path, experiment_folder,
             reverse_loss=True, image_name="test_elbo.png")


if __name__ == '__main__':
    experiment()

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

from prompt_dt.cvae.mnist.mnist_cvae import MnistCVAE
from prompt_dt.utils.other import seed_other
from prompt_dt.utils.path import *
from prompt_dt.utils.other import parse_config

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO

import tqdm

def get_train_data_loader(variant):
    dataset = MNIST(
        root='data', train=True, transform=transforms.ToTensor(),
        download=True)
    
    data_loader = DataLoader(
        dataset=dataset, batch_size=variant['batch_size'], shuffle=True)

    return data_loader

def experiment():
    # parse config file
    config_file_path = os.path.join(root_path, "prompt_dt", "cvae", "mnist", "config.yaml")
    variant = parse_config(config_file_path)
    # print out variant
    print("=========================== variant ==================================")
    print(variant)

    # seed
    seed_other(int(variant['seed']))

    # device
    device = variant['device']

    # train data loader
    train_data_loader = get_train_data_loader(variant)

    # create cvae net
    cvae_net = MnistCVAE(latent_dim=variant['latent_dim'], 
                    hidden_dim=variant['hidden_dim'], 
                    num_labels=10, 
                    prior_distribution=variant['prior_distribution'])
    cvae_net.to(device)

    # create optimizer
    optimizer = pyro.optim.Adam({"lr": variant['learning_rate']})

    # create ELBO optimizer
    svi = SVI(cvae_net.model, cvae_net.guide, optimizer, loss=Trace_ELBO())


    ts = time.time()
    
    # start training
    logs = defaultdict(list)
    num_epochs = int(variant['num_epochs'])
    print_every = int(variant["print_every"])
    for epoch in range(num_epochs):

        bar = tqdm(
            train_data_loader,
            desc="CVAE Epoch {}".format(epoch).ljust(20), # Left justify string of minimum width 20
        )
        
        # Iterate over the entire dataset once
        running_loss = 0.0
        for iteration, (x, c) in enumerate(bar):

            x, c = x.to(device), c.to(device)

            batch_size = x.size(0)
            batch_loss = svi.step(x, c)
            sample_loss = batch_loss / batch_size

            logs['loss'].append(sample_loss)
            running_loss += sample_loss

            if iteration % print_every == 0 or iteration == len(train_data_loader) - 1:
                # verbose
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, num_epochs, iteration, len(train_data_loader)-1, sample_loss))

                # generate data
                c = torch.arange(0, 10).long().unsqueeze(1).to(device)
                z = torch.randn([c.size(0), args.latent_size]).to(device)
                x = vae.inference(z, c=c)
            
                plt.figure()
                plt.figure(figsize=(5, 10))
                for p in range(10):
                    plt.subplot(5, 2, p+1)
                    
                    plt.text(
                        0, 0, "c={:d}".format(c[p].item()), color='black',
                        backgroundcolor='white', fontsize=8)
                    plt.imshow(x[p].view(28, 28).cpu().data.numpy())
                    plt.axis('off')

                if not os.path.exists(os.path.join(args.fig_root, str(ts))):
                    if not(os.path.exists(os.path.join(args.fig_root))):
                        os.mkdir(os.path.join(args.fig_root))
                    os.mkdir(os.path.join(args.fig_root, str(ts)))

                plt.savefig(
                    os.path.join(args.fig_root, str(ts),
                                 "E{:d}I{:d}.png".format(epoch, iteration)),
                    dpi=300)
                plt.clf()
                plt.close('all')

        df = pd.DataFrame.from_dict(tracker_epoch, orient='index')
        g = sns.lmplot(
            x='x', y='y', hue='label', data=df.groupby('label').head(100),
            fit_reg=False, legend=True)
        g.savefig(os.path.join(
            args.fig_root, str(ts), "E{:d}-Dist.png".format(epoch)),
            dpi=300)


if __name__ == '__main__':
    experiment()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
from collections import defaultdict
from prompt_dt.cvae.utils import set_up_experiment_name_folder, save_model, plot_llk
from prompt_dt.utils.other import seed_other
from prompt_dt.utils.path import *
from prompt_dt.utils.other import parse_config
from prompt_dt.cvae.data_loader import get_train_data_loader, get_test_data_loader
import numpy as np
from tqdm import tqdm

# c --> x
class DeterministicDecoder(nn.Module):
    def __init__(self, input_dim, output_dim,
                 hidden_dim):
        super().__init__()

        self.MLP = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, c):
        x = self.MLP(c)

        return x
    
    def generate_x(self, c):
        with torch.no_grad():
            pred_x = self.forward(c)
            
        return pred_x


def train_one_epoch(model, train_data_loader,
                    loss_fn, 
                    optimizer, device, logs,
                    epoch, variant, verbose=True):
        
        num_epochs = int(variant['num_epochs'])
        num_train_samples = len(train_data_loader.dataset)

        model.train()
        
        bar = tqdm(
            train_data_loader, 
            desc="NN Epoch {}".format(epoch).ljust(20)
        )

        epoch_loss = 0
        for iteration, (target_x, c) in enumerate(bar):
            target_x, c = target_x.to(device), c.to(device)
            batch_size = target_x.size(0)

            optimizer.zero_grad()

            pred_x = model(c)
            loss = loss_fn(pred_x, target_x)
                
            loss.backward()
            optimizer.step()

            
            batch_loss = loss.item() * batch_size
            epoch_loss += batch_loss

        epoch_loss /= float(num_train_samples)
        logs['loss'].append(epoch_loss)
        
        # verbose
        if verbose:
            print("Epoch {:02d}/{:02d}, train loss {:9.4f}".format(
                epoch, num_epochs, epoch_loss))

def test(model, test_data_loader,
         loss_fn, device,
         epoch, variant, verbose=True):
    
    num_epochs = int(variant['num_epochs'])

    num_test_samples = len(test_data_loader.dataset)

    model.eval() 

    test_loss = 0
    # compute the loss over the entire test set
    for iteration, (x, c) in enumerate(test_data_loader):
        target_x, c = x.to(device), c.to(device)
        batch_size = target_x.size(0)
        with torch.no_grad():
            pred_x = model(c)
            loss = loss_fn(pred_x, target_x)
            batch_loss = loss.item() * batch_size
            test_loss += batch_loss

    test_loss /= float(num_test_samples)

    # verbose
    if verbose:
        print('-'*80)
        print("Epoch {:02d}/{:02d}, test loss {:9.4f}".format(
            epoch, num_epochs, test_loss))
        print('-'*80)
    
    return test_loss
     
def experiment():
    # set variables
    variant = { "base_env": "cheetah_dir",  # ['cheetah_dir', 'cheetah_vel', 'ant_dir', 'ML1-pick-place-v2']
                "algorithm_name": "baseline",
                "seed": 1,
                "device": "cuda:1",
                "hidden_dim": 512,
                "num_epochs": 40,
                "batch_size": 256,
                "learning_rate": 0.001,
                "test_frequency": 5 # test every n epoch
                }
    
    # print out variant
    print("=========================== variant ==================================")
    print(variant)

    # seed
    seed = int(variant['seed'])
    seed_other(seed)

    # device
    device = variant['device']

    # train data loader
    train_data_loader = get_train_data_loader(variant)

    # test data loader
    test_data_loader = get_test_data_loader(variant)


    # create regression net
    model = DeterministicDecoder(num_labels=10, 
                    image_height=28, image_width=28,
                    hidden_dim=int(variant['hidden_dim']))
    model.to(device)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=variant['learning_rate'])
    
    # set up loss function
    loss_fn = nn.BCELoss()

    # set up experiment name
    experiment_name, experiment_folder = set_up_experiment_name_folder(variant, seed)

    # prepare training log
    logs = defaultdict(list)
    logs['loss'] = []

    # start training
    num_epochs = int(variant['num_epochs'])
    test_frequency = int(variant["test_frequency"])

    test_loss_list = []
    test_epoch_list = []
    best_test_loss = np.inf

    for epoch in range(num_epochs):
        # train for one epoch
        train_one_epoch(model, train_data_loader,
                    loss_fn, 
                    optimizer, device, logs,
                    epoch, variant,)
        
        # evaluate
        if epoch % test_frequency == 0 or epoch == num_epochs - 1:
            # evaluate over test dataset
            test_loss = test(model, test_data_loader,
                    loss_fn, device,
                    epoch, variant)
            test_loss_list.append(test_loss)
            test_epoch_list.append(epoch)

            # save best model until now
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                save_model(model=model, 
                           runs_path=mnist_runs_path,
                           experiment_folder=experiment_folder)
                print("Better performance achieved: %f, model saved"%(best_test_loss))
            
    # plot and save test elbo during the training process
    plot_llk(test_loss_list, test_epoch_list, 
             mnist_figure_path, experiment_folder,
             reverse_loss=False, image_name="test_loss.png")


if __name__ == '__main__':
    experiment()
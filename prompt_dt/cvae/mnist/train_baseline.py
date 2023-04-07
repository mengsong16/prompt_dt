import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
from collections import defaultdict
from prompt_dt.cvae.mnist.mnist_utils import get_train_data_loader, get_test_data_loader, save_generated_images
from prompt_dt.cvae.mnist.mnist_utils import idx2onehot
from prompt_dt.cvae.utils import set_up_experiment_name_folder, save_model, plot_llk
from prompt_dt.utils.other import seed_other
from prompt_dt.utils.path import *
from prompt_dt.utils.other import parse_config
import numpy as np
from tqdm import tqdm

# c --> x
class DeterministicImageDecoder(nn.Module):
    def __init__(self, num_labels, 
                 image_height, image_width,
                 hidden_dim):
        super().__init__()

        self.image_height = image_height
        self.image_width = image_width

        self.num_labels = num_labels

        data_dim = self.image_height * self.image_width

        self.MLP = nn.Sequential(
            nn.Linear(self.num_labels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, label_idx):
        c = idx2onehot(label_idx, n=self.num_labels)

        x = self.MLP(c)
        x = self.sigmoid(x)

        return x
    
    def generate_x(self, label_idx):
        with torch.no_grad():
            pred_x = self.forward(label_idx)
            # reshape 1D vectors to 2D images
            recon_x = pred_x.reshape(-1, self.image_height, self.image_width)
        
        return recon_x



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
        for iteration, (target_x,label) in enumerate(bar):
            target_x, label = target_x.to(device), label.to(device)
            batch_size = target_x.size(0)
            target_x = target_x.view(batch_size, -1)

            optimizer.zero_grad()

            pred_x = model(label)
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
            target_x = target_x.view(batch_size, -1)
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
    # parse config file
    config_file_path = os.path.join(mnist_path, "mnist_baseline_config.yaml")
    variant = parse_config(config_file_path)
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
    model = DeterministicImageDecoder(num_labels=10, 
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
            
            # generate data and save
            image_labels = torch.arange(0, 10).long().unsqueeze(1).to(device)
            generated_x = model.generate_x(image_labels)
            save_generated_images(image_labels, generated_x, epoch, experiment_folder)
    
    # plot and save test elbo during the training process
    plot_llk(test_loss_list, test_epoch_list, 
             mnist_figure_path, experiment_folder,
             reverse_loss=False, image_name="test_loss.png")


if __name__ == '__main__':
    experiment()
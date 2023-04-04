import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from prompt_dt.cvae.cvae import CVAE

# index to one hot vector
def idx2onehot(idx, n):
    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    return onehot


class MnistCVAE(CVAE):
    def __init__(self, latent_dim, hidden_dim, num_labels, 
                 prior_distribution):
        
        self.num_labels = num_labels
        self.condition_dim = num_labels
        self.data_dim = 28 * 28
        
        super().__init__(self.data_dim, latent_dim, self.condition_dim, 
                         hidden_dim, prior_distribution, 
                         decoder_distribution="bernoulli")
        
    def guide(self, x, label_idx):
        # flatten 28*28 2D image to 1D vector
        if x.dim() > 2:
            x = x.view(-1, self.data_dim)
        
        c = idx2onehot(label_idx, n=self.num_labels)

        return super().guide(x, c)

    def model(self, label_idx):
        c = idx2onehot(label_idx, n=self.num_labels)
        
        return super().model(c)

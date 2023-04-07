import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from prompt_dt.cvae.cvae import CVAE
from prompt_dt.cvae.mnist.mnist_utils import idx2onehot



class ImageCVAE(CVAE):
    def __init__(self, latent_dim, hidden_dim, num_labels, 
                 image_height, image_width,
                 prior_distribution):
        
        self.num_labels = num_labels
        self.condition_dim = num_labels
        self.image_height = image_height
        self.image_width = image_width
        self.data_dim = self.image_height * self.image_width
        
        super().__init__(self.data_dim, latent_dim, self.condition_dim, 
                         hidden_dim, prior_distribution, 
                         decoder_distribution="bernoulli")
        
    def guide(self, x, label_idx, beta):
        # flatten 28*28 2D images to 1D vectors
        if x.dim() > 2:
            x = x.view(-1, self.data_dim)
        
        c = idx2onehot(label_idx, n=self.num_labels)

        return super().guide(x, c, beta)

    def model(self, x, label_idx, beta):
        c = idx2onehot(label_idx, n=self.num_labels)
        
        return super().model(x, c, beta)

    def reconstruct_x(self, x, label_idx, deterministic):
        # flatten 28*28 2D images to 1D vectors
        if x.dim() > 2:
            x = x.view(-1, self.data_dim)
        
        c = idx2onehot(label_idx, n=self.num_labels)
        
        recon_x = super().reconstruct_x(x, c, deterministic)
       
        # reshape 1D vectors to 2D images
        recon_x = recon_x.reshape(-1, self.image_height, self.image_width)
        
        return recon_x
    
    def embed_x(self, x, label_idx):
        # flatten 28*28 2D images to 1D vectors
        if x.dim() > 2:
            x = x.view(-1, self.data_dim)
        
        c = idx2onehot(label_idx, n=self.num_labels)

        return super().embed_x(x, c)
    

    def generate_x(self, label_idx, deterministic):
        c = idx2onehot(label_idx, n=self.num_labels)

        generated_x = super().generate_x(c, deterministic)

        # reshape 1D vectors to 2D images
        generated_x = generated_x.reshape(-1, self.image_height, self.image_width)

        return generated_x
        




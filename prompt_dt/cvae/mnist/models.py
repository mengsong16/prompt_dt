import torch
import torch.nn as nn
from torch.distributions import Normal

# index to one hot vector
def idx2onehot(idx, n):
    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    return onehot


class VAE(nn.Module):
    def __init__(self, data_dim, latent_dim, condition_dim, hidden_dim,
                 conditional, prior_distribution):
        super().__init__()

        self.latent_dim = latent_dim
        self.conditional = conditional
        self.prior_distribution = prior_distribution

        self.encoder = StochasticEncoder(
            data_dim, latent_dim, hidden_dim, conditional, condition_dim)
        self.decoder = StochasticDecoder(
            data_dim, latent_dim, hidden_dim, conditional, condition_dim)

    def forward(self, x, c=None):
        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, c=None):
        recon_x = self.decoder(z, c)

        return recon_x
    
    def generate_latent_variables(self, batch_size, c=None):
        if self.prior_distribution == "standard_gaussian":
            z = torch.randn([batch_size, self.latent_dim])
        if self.prior_distribution == "gaussian_identity_variance":
            z = torch.randn([batch_size, self.latent_dim])

        else:
            print("Error: undefined prior distribution")
            exit()

        return z
    
    def generate_data(self, batch_size, c=None):
        z = self.generate_latent_variables(batch_size, c)
        if self.conditional:
            recon_x = self.inference(z, c)
        else:
            recon_x = self.inference(z)
        
        return recon_x


class MnistVAE(VAE):
    def __init__(self, latent_dim, hidden_dim,
                 conditional, num_labels):
        
        self.num_labels = num_labels
        self.condition_dim = num_labels
        self.conditional = conditional
        self.data_dim = 28 * 28
        
        super().__init__(self.data_dim, latent_dim, self.condition_dim, 
                         hidden_dim, conditional)
        
    def forward(self, x, label_idx=None):
        # flatten 28*28 2D image to 1D vector
        if x.dim() > 2:
            x = x.view(-1, self.data_dim)
        
        if self.conditional:
            label_idx = idx2onehot(label_idx, n=self.num_labels)

        return super().forward(x, label_idx)

    def inference(self, z, label_idx=None):
        if self.conditional:
            label_idx = idx2onehot(label_idx, n=self.num_labels)
        
        return super().inference(z, label_idx)
    
    

# x --> z
# x + c --> z
class StochasticEncoder(nn.Module):
    def __init__(self, data_dim, latent_dim, hidden_dim, conditional, condition_dim):
        super().__init__()

        self.conditional = conditional
        
        input_dim = data_dim
        if self.conditional:
            input_dim += condition_dim
            
        self.MLP = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Normal distribution
        self.linear_means = nn.Linear(hidden_dim, latent_dim)
        self.linear_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, c=None):
        if self.conditional:
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars

# z --> x
# z + c --> x
class StochasticDecoder(nn.Module):
    def __init__(self, data_dim, latent_dim, hidden_dim, conditional, condition_dim):
        super().__init__()

        self.conditional = conditional

        input_dim = latent_dim
        if self.conditional:
            input_dim += condition_dim
        
        self.MLP = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, data_dim),
            nn.Sigmoid(),  # Bernoulli distribution
        )

    def forward(self, z, c=None):
        if self.conditional:
            z = torch.cat((z, c), dim=-1)

        z = self.MLP(z)
        x = self.output_head(z)

        return x

# c --> x
class DeterministicDecoder(nn.Module):
    def __init__(self, condition_dim, data_dim, hidden_dim):
        super().__init__()

        self.MLP = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def forward(self, c):
        x = self.MLP(c)

        return x
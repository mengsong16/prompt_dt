import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

class CVAE(nn.Module):
    def __init__(self, data_dim, latent_dim, condition_dim, hidden_dim,
                 prior_distribution, decoder_distribution):
        super().__init__()

        self.latent_dim = latent_dim
        self.prior_distribution = prior_distribution
        self.decoder_distribution = decoder_distribution

        self.encoder = StochasticEncoder(
            data_dim, latent_dim, hidden_dim, condition_dim)
        self.decoder = StochasticDecoder(
            data_dim, latent_dim, hidden_dim, condition_dim,
            self.decoder_distribution)
        self.prior_net = Prior(condition_dim, latent_dim, hidden_dim, 
                           mean_only=True)
    
        # register the parameters of these torch.nn.Modules with the ParamStore
        pyro.module("encoder", self.encoder)
        pyro.module("decoder", self.decoder)
        pyro.module("prior_net", self.prior_net)
    
    # define the model p(x|z,c)p(z|c) [decoder and prior_net]
    # used by svi to compute p(x,z) in ELBO
    def model(self, x, c, beta):
        # inform Pyro that the variables in the batch of (c) are conditionally independent
        with pyro.plate("data"):
            batch_size = c.shape[0]
            identity_vars = torch.ones(batch_size, self.latent_dim, dtype=c.dtype, device=c.device)
            # define p(z|c) and sample a batch of z from it
            if self.prior_distribution == "gaussian_identity_covariance":
                # Gaussian with identity covariance matrix: N(mu, I)
                prior_means = self.prior_net(c)
                with poutine.scale(scale=beta):
                    z = pyro.sample("latent", dist.Normal(prior_means, identity_vars).to_event(1))
            elif self.prior_distribution == "standard_gaussian":
                # Standard Gaussian: N(0, I)
                zero_means = torch.zeros(batch_size, self.latent_dim, dtype=c.dtype, device=c.device)
                with poutine.scale(scale=beta):
                    z = pyro.sample("latent", dist.Normal(zero_means, identity_vars).to_event(1))
            else:
                print("Error: Undefined prior distribution")
                exit()

            # define p(x|z,c) and sample a batch of x from it
            if self.decoder_distribution == "bernoulli":
                # Bernoulli(loc)
                loc = self.decoder(z, c)
                pyro.sample(
                    "obs",
                    dist.Bernoulli(loc, validate_args=False).to_event(1),
                )
            elif self.decoder_distribution == "diagonal_gaussian":
                means, vars = self.decoder(z, c)
                pyro.sample(
                    "obs", 
                    dist.Normal(means, vars).to_event(1)
                )
            else:
                print("Error: Undefined decoder distribution")
                exit()

    # define the guide q(z|x,c) [encoder]
    # used by svi to compute q(z|x,c) in ELBO
    def guide(self, x, c, beta):
        # inform Pyro that the variables in the batch of (x, c) are conditionally independent
        with pyro.plate("data"):
            # define q(z|x,c) and sample a batch of z from it
            # Diagonal Gaussian: N(mu, diag(vars))
            means, vars = self.encoder(x, c)
            with poutine.scale(scale=beta):
                pyro.sample("latent", dist.Normal(means, vars).to_event(1))

    # get deterministic embedding of x
    def embed_x(self, x, c):
        # encode x
        means, _ = self.encoder(x, c)

        return means

    # decode z
    def decode_z(self, z, c, deterministic):
        if self.decoder_distribution == "bernoulli":
            # Bernoulli(loc)
            loc = self.decoder(z, c)
            if deterministic:
                recon_x = loc
            else:
                # sample in x space
                recon_x = dist.Bernoulli(loc, validate_args=False).to_event(1).sample()
        elif self.decoder_distribution == "diagonal_gaussian":
            means, vars = self.decoder(z, c)
            if deterministic:
                recon_x = means
            else:
                # sample in x space
                recon_x = dist.Normal(means, vars).to_event(1).sample()
        else:
            print("Error: Undefined decoder distribution")
            exit()
        
        return recon_x
    
    # x: [B, obs_dim]
    # recon_x: [B, obs_dim]
    # reconstruct given x
    def reconstruct_x(self, x, c, deterministic):
        # encode x
        means, vars = self.encoder(x, c)
        # sample in z space
        z = dist.Normal(means, vars).to_event(1).sample()
        # decode z
        recon_x = self.decode_z(z, c, deterministic)
        
        return recon_x

    # sample z from prior
    def sample_z(self, c):
        batch_size = c.shape[0]
        identity_vars = torch.ones(batch_size, self.latent_dim, dtype=c.dtype, device=c.device)
        # Sample a batch of z from p(z|c)
        if self.prior_distribution == "gaussian_identity_covariance":
            # Gaussian with identity covariance matrix: N(mu, I)
            prior_means = self.prior_net(c)
            z = dist.Normal(prior_means, identity_vars).to_event(1).sample()
        elif self.prior_distribution == "standard_gaussian":
            # Standard Gaussian: N(0, I)
            zero_means = torch.zeros(batch_size, self.latent_dim, dtype=c.dtype, device=c.device)
            z = dist.Normal(zero_means, identity_vars).to_event(1).sample()
        else:
            print("Error: Undefined prior distribution")
            exit()
        
        return z
    
    # generate x from prior
    def generate_x(self, c, deterministic):
        # sample z from prior
        z = self.sample_z(c)
        # decode z
        generated_x = self.decode_z(z, c, deterministic)

        return generated_x


# x + c --> z
class StochasticEncoder(nn.Module):
    def __init__(self, data_dim, latent_dim, hidden_dim, 
                condition_dim):
        super().__init__()

        self.MLP = nn.Sequential(
            nn.Linear(data_dim+condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Diagonal Normal distribution
        self.linear_means = nn.Linear(hidden_dim, latent_dim)
        self.linear_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, c):
        x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        vars = torch.exp(log_vars)

        return means, vars

# c --> z
class Prior(nn.Module):
    def __init__(self, condition_dim, latent_dim, hidden_dim, mean_only):
        super().__init__()

        self.mean_only = mean_only
            
        self.MLP = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Normal distribution
        self.linear_means = nn.Linear(hidden_dim, latent_dim)
        if not self.mean_only:
            self.linear_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, c):
        z = self.MLP(c)
        means = self.linear_means(z)

        # Diagonal Normal distribution
        if not self.mean_only:
            log_vars = self.linear_log_var(z)
            vars = torch.exp(log_vars)
            return means, vars
        # Normal distribution with identity covariance matrix
        else:
            return means
        
# z + c --> x
class StochasticDecoder(nn.Module):
    def __init__(self, data_dim, latent_dim, hidden_dim, condition_dim,
                 decoder_distribution):
        super().__init__()

        self.decoder_distribution = decoder_distribution

        self.MLP = nn.Sequential(
            nn.Linear(latent_dim+condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        if self.decoder_distribution == "bernoulli":
            self.loc = nn.Sequential(
                nn.Linear(hidden_dim, data_dim),
                nn.Sigmoid(),  # Bernoulli distribution
            )
        elif self.decoder_distribution == "diagonal_gaussian":
            self.linear_means = nn.Linear(hidden_dim, latent_dim)
            self.linear_log_var = nn.Linear(hidden_dim, latent_dim)
        else:
            print("Error: Undefined decoder distribution")
            exit()

    def forward(self, z, c):
        z = torch.cat((z, c), dim=-1)

        z = self.MLP(z)

        if self.decoder_distribution == "bernoulli":
            return self.loc(z)
        elif self.decoder_distribution == "diagonal_gaussian":
            means = self.linear_means(z)
            log_vars = self.linear_log_var(z)
            vars = torch.exp(log_vars)

            return means, vars
        else:
            print("Error: Undefined decoder distribution")
            exit()

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
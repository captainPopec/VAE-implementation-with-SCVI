from typing import Literal
import torch
from scvi import REGISTRY_KEYS
from scvi.module.base import (
    BaseModuleClass,
    LossOutput,
    auto_move_data,
)
from torch.distributions import NegativeBinomial, Normal
from torch.distributions import kl_divergence as kl
import scvi


class My_Encoder(torch.nn.Module):
    def __init__(self, n_input: int, n_hidden: int, n_output: int, link_var: Literal["exp", "none", "softmax"]):  

        super().__init__()
        self.neural_net = torch.nn.Sequential(
            torch.nn.Linear(n_input, n_hidden), # if n_hidden is a power of 2 GPU handles it quicker
            torch.nn.BatchNorm1d(n_hidden),   # normalizira izlaz iz linearne transformacije tako da ima srednju vrijednost 0 i standardnu devijaciju 1, sto pomaze u stabilizaciji i ubrzanju treniranja neuralne mreze - no moving goalpost
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_output),
        )
        
        self.transformation = None #pazi ouaj kurac se primjenjujej tak na kraju, ne između svakog layera
        if link_var == "exp":
            self.transformation = torch.exp
        elif link_var == "softmax":
            self.transformation = torch.nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        output = self.neural_net(x)    # dakle tu se poziva ne neural net koji je upravo definiro gore sa troch.nn.Sequential, i on ce vratiti output koji je linearna transformacija ulaza, a onda se na taj output primjenjuje transformacija koja je odabrana u konstruktoru klase (exp, softmax ili none)
        if self.transformation:
            output = self.transformation(output)
        return output



class My_Decoder(torch.nn.Module):
    def __init__(self, n_input: int, n_output: int, link_var: Literal["exp", "none", "softmax"]):

        super().__init__()
        self.neural_net = torch.nn.Sequential(
            torch.nn.Linear(n_input, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_output),
        )
        
        self.transformation = None
        if link_var == "exp":
            self.transformation = torch.exp
        elif link_var == "softmax":
            self.transformation = torch.nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        output = self.neural_net(x)    # dakle tu se poziva ne neural net koji je upravo definiro gore sa troch.nn.Sequential, i on ce vratiti output koji je linearna transformacija ulaza, a onda se na taj output primjenjuje transformacija koja je odabrana u konstruktoru klase (exp, softmax ili none)
        if self.transformation:
            output = self.transformation(output)
        return output



class My_VAE(BaseModuleClass):
    def __init__(
            self,
            n_input: int,
            n_hidden: int,
            n_latent: int,
            n_batch: int,
    ):
        super().__init__()
        
        # define decoder as neural network
        self.decoder = My_Decoder(n_latent, n_input, "softmax") # output means of each gene needs to be non-negative
        self.log_theta = torch.nn.Parameter(torch.randn(n_input)) # this is variance essentially of the output, so for each gene/protein
        #if we predicts something, it wont be optimized necceseraly, so I need to specify that theta is a parameter that needs to be optimized, so I use torch.nn.Parameter, and I initialize it with random values from a normal distribution with mean 0 and standard deviation 1, and the shape of this parameter is (n_input,), which means that we have one variance for each gene/protein in the input data
        # what I DON'T like is that this Theta does not emerge from the Neural Nets themselves, it is just a random number that is optimized

        #defining encoder
        self.mean_encoder = My_Encoder(n_input, n_hidden, n_latent, "none") # mean of latent var is ok if it is NEGATIVE, it just needs to make sense to the computer
        self.var_encoder = My_Encoder(n_input, n_hidden, n_latent, "exp") # we want variance to be positive, so we use exp as link function

        self.n_batch = n_batch

    def _get_inference_input(self, tensors):
        x = tensors[REGISTRY_KEYS.X_KEY]
        input_dict = dict(x=x)
        return input_dict

    @auto_move_data
    def inference(self, tensors):
        x = tensors[REGISTRY_KEYS.X_KEY]
        x_ = torch.log(x + 1) #logp1 essentially
        qz_m = self.mean_encoder(x_)
        qz_v = self.var_encoder(x_)
        #reparametrization trick
        z = Normal(qz_m, torch.sqrt(qz_v)).rsample() 

        outputs = dict(qz_m=qz_m, qz_v=qz_v, z=z)
        return outputs
    
    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        x = tensors[REGISTRY_KEYS.X_KEY]
        library_size = torch.sum(x, dim=1, keepdim=True) # library size is sum of counts for each cell, so we sum across genes/proteins for each cell

        input_dict = {
            "z": z,
            "library_size": library_size
        }
        return input_dict


    @auto_move_data
    def generative(self, z, library_size):
        
        # produces normalized MEAN of the negative binomial - it doesnt really calculate it in a structured way, it just a softmax output of NN
        px_scale = self.decoder(z) 
        # we need to multiply by library size to get the mean of the negative binomial, because the mean of the negative binomial is the product of the normalized mean and the library size
        px_rate = px_scale * library_size
        # theta is the variance of the negative binomial, and it is a parameter that is optimized during training, it is not a function of the input data, it is just a random number that is optimized during training
        theta = torch.exp(self.log_theta) 

        '''
        Shvati da je px_scale jednsotavno array upravo generairan decoderom, možeš ju množit, radit što hoćeš
        Upravo na isti način generiramo i thetu, prije smo definirali log_theta kao nešto što proizvodi array brojki.
        
        Dakle ova f-ja generative() i inferance() samo vrše računanje s funkcijama definiranim u init() - bitno je da skužiš jednostavnost koncepcije
        
        KOliko shvaćam mi ništa nismo logaritmirali, ali koristimo to u eskponentu pa se očekuje da je to vrijednost kojom potenciramo log thateta ako je ciljani rezultat potencijajcije theta
        A ona priča o pozitivnosti je vrlo jednsotavna, e^x > 0 za svaki x element R
        '''

        return dict(px_scale=px_scale, px_rate=px_rate, theta=theta) # px stands for "predicted x"
    
    def loss(self, tensors, inference_outputs, generative_outputs):
        x = tensors[REGISTRY_KEYS.X_KEY]
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        px_rate = generative_outputs["px_rate"]
        theta = generative_outputs["theta"]

        
        # Calculate likelihood loss
        nb_logits = (px_rate + 1e-8).log() - (theta + 1e-8).log() # log of mean minus log of variance, this is the logit parameter of the negative binomial distribution
        log_likelihood = NegativeBinomial(total_count=theta, logits=nb_logits).log_prob(x).sum(dim=-1) # -1 je jer je to column za GENE, tj sumira preko svih gena #nisam bas skužio zašto je total counts = theta, ali uprincipu to je dipersion
        #ovdje smo izračunali sumu svih log_likelihooda, a likelihhodi odgovaraju na pitanje; koja je šansa da izvučemo x iz predviđene ditribucije

        #calculate KL Regularization
        prior_dist = Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v)) # prior distribution is standard normal
        post_dist = Normal(qz_m, torch.sqrt(qz_v)) # variational distribution is normal with mean and variance predicted by encoder
        kl_divergence = kl(post_dist, prior_dist).sum(dim=-1)

        elbo = log_likelihood - kl_divergence
        loss = torch.mean(-elbo) # we want to maximize elbo
        '''
        we obviously take MEAN across all cells, PAZI elbo NIJE jedan broj nego matrica n_cells x 1, 
        1 jer se za gene već sumiralo vrijednossti posteriora i KL preko latents dakle imali smo log_likelihood(n_cells x n_genes).sum
        te KL(n_cells x n_latent).sum
        Tad smo računali SUMU jer ...??
        
        '''
        return LossOutput(loss=loss, recon_loss=-torch.mean(log_likelihood), kl_loss=torch.mean(kl_divergence))

    
'''
--------------------------------------- Defining base SCVI model ----------------------------------------------------------------------------------------------
'''

class SCVI_NeuralNet(torch.nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        link_var: Literal["exp", "none", "softmax"],
    ):
        """
        Encodes data of ``n_input`` dimensions into a space of ``n_output`` dimensions.

        Uses a one layer fully-connected neural network with 128 hidden nodes.

        Parameters
        ----------
        n_input
            The dimensionality of the input
        n_output
            The dimensionality of the output
        link_var
            The final non-linearity
        """
        super().__init__()
        self.neural_net = torch.nn.Sequential(
            torch.nn.Linear(n_input, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_output),
        )
        self.transformation = None
        if link_var == "softmax":
            self.transformation = torch.nn.Softmax(dim=-1)
        elif link_var == "exp":
            self.transformation = torch.exp

    def forward(self, x: torch.Tensor):
        output = self.neural_net(x)
        if self.transformation:
            output = self.transformation(output)
        return output


class SCVI_VAE(BaseModuleClass):
    """
    Skeleton Variational auto-encoder model.

    Here we implement a basic version of scVI's underlying VAE [Lopez18]_.
    This implementation is for instructional purposes only.

    Parameters
    ----------
    n_input
        Number of input genes
    n_latent
        Dimensionality of the latent space
    """

    def __init__(
        self,
        n_input: int,
        n_latent: int = 10,
    ):
        super().__init__()
        # in the init, we create the parameters of our elementary stochastic computation unit.

        # First, we setup the parameters of the generative model
        self.decoder = SCVI_NeuralNet(n_latent, n_input, "softmax")
        self.log_theta = torch.nn.Parameter(torch.randn(n_input))

        # Second, we setup the parameters of the variational distribution
        self.mean_encoder = SCVI_NeuralNet(n_input, n_latent, "none")
        self.var_encoder = SCVI_NeuralNet(n_input, n_latent, "exp")

    def _get_inference_input(self, tensors):
        """Parse the dictionary to get appropriate args"""
        # let us fetch the raw counts, and add them to the dictionary
        x = tensors[REGISTRY_KEYS.X_KEY]

        input_dict = dict(x=x)
        return input_dict

    @auto_move_data
    def inference(self, x):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        # log the input to the variational distribution for numerical stability
        x_ = torch.log(1 + x)
        # get variational parameters via the encoder networks
        qz_m = self.mean_encoder(x_)
        qz_v = self.var_encoder(x_)
        # get one sample to feed to the generative model
        # under the hood here is the Reparametrization trick (Rsample)
        z = Normal(qz_m, torch.sqrt(qz_v)).rsample()

        outputs = dict(qz_m=qz_m, qz_v=qz_v, z=z)
        return outputs

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        x = tensors[REGISTRY_KEYS.X_KEY]
        # here we extract the number of UMIs per cell as a known quantity
        library = torch.sum(x, dim=1, keepdim=True)

        input_dict = {
            "z": z,
            "library": library,
        }
        return input_dict

    @auto_move_data
    def generative(self, z, library):
        """Runs the generative model."""

        # get the "normalized" mean of the negative binomial
        px_scale = self.decoder(z)
        # get the mean of the negative binomial
        px_rate = library * px_scale
        # get the dispersion parameter
        theta = torch.exp(self.log_theta)

        return dict(px_scale=px_scale, theta=theta, px_rate=px_rate)

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
    ):
        # here, we would like to form the ELBO. There are two terms:
        #   1. one that pertains to the likelihood of the data
        #   2. one that pertains to the variational distribution
        # so we extract all the required information
        x = tensors[REGISTRY_KEYS.X_KEY]
        px_rate = generative_outputs["px_rate"]
        theta = generative_outputs["theta"]
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]

        # term 1
        # the pytorch NB distribution uses a different parameterization
        # so we must apply a quick transformation (included in scvi-tools, but here we use the pytorch code)
        nb_logits = (px_rate + 1e-8).log() - (theta + 1e-8).log()
        log_lik = (
            NegativeBinomial(total_count=theta, logits=nb_logits)
            .log_prob(x)
            .sum(dim=-1)
        )

        # term 2
        prior_dist = Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v))
        var_post_dist = Normal(qz_m, torch.sqrt(qz_v))
        kl_divergence = kl(var_post_dist, prior_dist).sum(dim=1)

        elbo = log_lik - kl_divergence
        loss = torch.mean(-elbo)
        return LossOutput(loss=loss, recon_loss=-torch.mean(log_lik), kl_loss=torch.mean(kl_divergence))
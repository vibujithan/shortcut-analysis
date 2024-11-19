import monai
import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
from monai.networks.blocks import Convolution
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Act, Norm

from utils.modules import CMADE


class SFCN(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(monai.networks.blocks.Convolution(3, 1, 2, strides=1, kernel_size=3),
                                    nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3)))
        self.block2 = nn.Sequential(monai.networks.blocks.Convolution(3, 2, 4, strides=1, kernel_size=3),
                                    nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3)))
        self.block3 = nn.Sequential(monai.networks.blocks.Convolution(3, 4, 8, strides=1, kernel_size=3),
                                    nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3)))
        self.block4 = nn.Sequential(monai.networks.blocks.Convolution(3, 8, 8, strides=1, kernel_size=3),
                                    nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.block5 = nn.Sequential(monai.networks.blocks.Convolution(3, 8, 16, strides=1, kernel_size=3),
                                    nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.block5_1 = nn.Sequential(monai.networks.blocks.Convolution(3, 16, 16, strides=1, kernel_size=3),
                                      nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.block6 = monai.networks.blocks.Convolution(3, 16, 16, strides=1, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

        # Block 7
        self.avgpool1 = nn.AvgPool3d(kernel_size=(1, 1, 1))
        self.dropout1 = nn.Dropout(.5)
        self.flat1 = nn.Flatten()
        self.fc1 = nn.Linear(16, 1)
        # self.fc2 = nn.Linear(10, 1)
        self.block7 = nn.Sequential(self.avgpool1, self.dropout1, self.flat1, self.fc1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        # x = self.block6(x)
        x = self.block7(x)
        # x = self.sigmoid(x)

        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_size=512):
        super().__init__()
        self.latent_size = latent_size

        self.encoder_layers = nn.Sequential(
            # Layer 1: 180x180x180 -> 90x90x90
            Convolution(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                strides=2,
                padding=0,
                act=Act.RELU,
                norm=Norm.BATCH
            ),

            # Layer 2: 90x90x90 -> 45x45x45
            Convolution(
                spatial_dims=3,
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                strides=2,
                padding=1,
                act=Act.RELU,
                norm=Norm.BATCH
            ),

            # Layer 3: 45x45x45 -> 23x23x23
            Convolution(
                spatial_dims=3,
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                strides=2,
                padding=0,
                act=Act.RELU,
                norm=Norm.BATCH
            ),

            # Layer 4: 23x23x23 -> 12x12x12
            Convolution(
                spatial_dims=3,
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                strides=2,
                padding=1,
                act=Act.RELU,
                norm=Norm.BATCH
            ),
        )

        # Flatten and project to latent space
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 11 * 11 * 11, latent_size)

    def forward(self, x):
        x = self.encoder_layers(x)
        x = self.flatten(x)
        return self.fc(x)

    def get_output_size(self):
        return self.latent_size


class Decoder(nn.Module):
    def __init__(self, latent_size=512, out_channels=1):
        super().__init__()

        self.fc = nn.Linear(latent_size, 256 * 11 * 11 * 11)
        self.unflatten = nn.Unflatten(1, (256, 11, 11, 11))

        self.decoder_layers = nn.Sequential(
            # Layer 1: 12x12x12 -> 23x23x23
            UpSample(
                spatial_dims=3,
                in_channels=256,
                out_channels=128,
                scale_factor=2,
                mode='deconv'
            ),
            Convolution(
                spatial_dims=3,
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1,
                act=Act.RELU,
                norm=Norm.BATCH
            ),

            # Layer 2: 23x23x23 -> 45x45x45
            UpSample(
                spatial_dims=3,
                in_channels=128,
                out_channels=64,
                scale_factor=2,
                mode='deconv'
            ),
            Convolution(
                spatial_dims=3,
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                act=Act.RELU,
                norm=Norm.BATCH
            ),

            # Layer 3: 45x45x45 -> 90x90x90
            UpSample(
                spatial_dims=3,
                in_channels=64,
                out_channels=32,
                scale_factor=2,
                mode='deconv'
            ),
            Convolution(
                spatial_dims=3,
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=2,
                act=Act.RELU,
                norm=Norm.BATCH
            ),

            # Layer 4: 90x90x90 -> 180x180x180
            UpSample(
                spatial_dims=3,
                in_channels=32,
                out_channels=32,
                scale_factor=2,
                mode='deconv'
            ),
            Convolution(
                spatial_dims=3,
                in_channels=32,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                act=None,
                norm=None
            )
        )

    def forward(self, z):
        x = self.fc(z)
        x = self.unflatten(x)
        x = self.decoder_layers(x)
        return x


class Autoencoder3D(nn.Module):
    def __init__(self, in_channels=1, latent_size=512):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, latent_size=latent_size)
        self.decoder = Decoder(latent_size=latent_size, out_channels=in_channels)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class Flow(nn.Module):
    """ Masked Causal Flow that uses a MADE-style network for fast-forward """

    def __init__(self, dim, edges, device, net_class=CMADE, hm=[4, 6, 4]):
        super().__init__()
        self.dim = dim
        self.net = net_class(dim, dim * 2, edges, hm)
        self.device = device

    def forward(self, x):
        # here we see that we are evaluating all of z in parallel, so density estimation will be fast
        st = torch.nan_to_num(self.net(x), nan=0.0, posinf=1e3, neginf=-1e3)
        s, t = st.split(self.dim, dim=1)

        z = x * torch.exp(s) + t
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def backward(self, z):
        # we have to decode the x one at a time, sequentially
        x = torch.zeros_like(z).to(self.device)
        log_det = torch.zeros(z.size(0)).to(self.device)
        for i in range(self.dim):
            st = torch.nan_to_num(self.net(x), nan=0.0, posinf=1e3, neginf=-1e3)
            s, t = st.split(self.dim, dim=1)
            x[:, i] = (z[:, i] - t[:, i]) * torch.exp(-s[:, i])
            log_det += -s[:, i]
        return x, log_det


class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m).to(x.device)
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m).to(z.device)
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z)
            log_det += ld
            xs.append(z)
        return xs, log_det


class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """

    def __init__(self, priors, flows):
        super().__init__()
        self.priors = priors
        self.flow = NormalizingFlow(flows)

    def forward(self, x):
        zs, log_det = self.flow.forward(x)
        if type(self.priors) == list:
            prior_log_prob = 0
            for sl, dist in self.priors:
                data = zs[-1][:, sl]
                prior_log_prob += dist.log_prob(data).view(x.size(0), -1).sum(1)
        else:
            prior_log_prob = self.priors.log_prob(zs[-1]).view(x.size(0), -1).sum(1)

        return zs, prior_log_prob, log_det

    def backward(self, z):
        xs, log_det = self.flow.backward(z)
        return xs, log_det

    def sample(self, num_samples):
        z = self.priors.sample((num_samples,))
        xs, _ = self.flow.backward(z)
        return xs

    def log_likelihood(self, x):
        if type(x) is np.ndarray:
            x = torch.tensor(x.astype(np.float32))
        _, prior_logprob, log_det = self.forward(x)
        return (prior_logprob + log_det).cpu().detach().numpy()


class MACAW(nn.Module):
    def __init__(self, nlatents, nlayers=4, hidden=[4, 6, 4], device='cuda'):
        super().__init__()

        self.nlatents = nlatents
        self.n_layers = nlayers
        self.hidden = hidden
        self.device = device
        self.ncauses = 4

        P_PD = 0.35
        P_sex = 0.61
        P_study = np.array([18.702290076335878,
                            7.175572519083969,
                            3.2061068702290076,
                            2.8244274809160306,
                            3.435114503816794,
                            4.732824427480916,
                            30.763358778625953,
                            5.648854961832061,
                            9.236641221374047,
                            5.9541984732824424,
                            5.801526717557252,
                            2.519083969465649]) / 100

        P_scanner_type = np.array([24.35114503816794,
                                   7.251908396946565,
                                   3.2061068702290076,
                                   8.778625954198473,
                                   3.435114503816794,
                                   31.14503816793893,
                                   4.2748091603053435,
                                   9.236641221374047,
                                   5.801526717557252,
                                   2.519083969465649]) / 100

        # Causal DAG
        study_to_latents = [(0, i) for i in range(self.ncauses, self.nlatents + self.ncauses)]
        sex_to_latents = [(1, i) for i in range(self.ncauses, self.nlatents + self.ncauses)]
        scanner_to_latents = [(2, i) for i in range(self.ncauses, self.nlatents + self.ncauses)]
        PD_to_latents = [(3, i) for i in range(self.ncauses, self.nlatents + self.ncauses)]

        # study_to_sex = [(0, 1)]
        study_to_scanner = [(0, 2)]
        # study_to_pd = [(0, 3)]

        # sex_to_pd = [(1, 3)]

        autoregressive_latents = [(i, j) for i in range(self.ncauses, self.nlatents + self.ncauses) for j in
                                  range(i + 1, self.nlatents + self.ncauses)]

        edges = (study_to_latents +
                 sex_to_latents +
                 scanner_to_latents +
                 PD_to_latents +
                 study_to_scanner +
                 autoregressive_latents)

        priors = [
            (slice(0, 1), td.Categorical(torch.tensor(P_study).to(self.device))),
            (slice(1, 2), td.Bernoulli(torch.tensor(P_sex).to(self.device))),
            (slice(2, 3), td.Normal(torch.zeros(1).to(self.device), torch.ones(
                1).to(self.device))),
            (slice(3, 4), td.Bernoulli(torch.tensor(P_PD).to(self.device))),
            (slice(self.ncauses, self.nlatents + self.ncauses),
             td.Normal(torch.zeros(self.nlatents).to(self.device), torch.ones(
                 self.nlatents).to(self.device)))]

        flow_list = [Flow(self.nlatents + self.ncauses, edges, self.device, hm=hidden) for _ in range(nlayers)]

        self.model = NormalizingFlowModel(priors, flow_list)

    def forward(self, batch):
        zs, prior_log_prob, log_det = self.model(batch)
        return zs, prior_log_prob, log_det

    def backward(self, z):
        return self.model.backward(z)

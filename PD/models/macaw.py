import networkx as nx
import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F


class MACAW(nn.Module):
    def __init__(self, nlatents, nlayers=6, hidden=[4, 6, 4], device="cuda"):
        super().__init__()

        self.nlatents = nlatents
        self.n_layers = nlayers
        self.hidden = hidden
        self.device = device
        self.ncauses = 9 + 8 + 1 + 1

        P_PD = 0.5101796407185629
        P_sex = 0.57724550
        P_study = np.array(
            [
                0.14491017964071856,
                0.2934131736526946,
                0.13532934131736526,
                0.09101796407185629,
                0.05029940119760479,
                0.05389221556886228,
                0.07425149700598803,
                0.04431137724550898,
                0.1125748502994012,
            ]
        )

        # Causal DAG
        study_to_latents = [
            (i, j)
            for i in range(9)
            for j in range(self.ncauses, self.nlatents + self.ncauses)
        ]
        sex_to_latents = [
            (9, i) for i in range(self.ncauses, self.nlatents + self.ncauses)
        ]
        scanner_to_latents = [
            (10 + i, j)
            for i in range(8)
            for j in range(self.ncauses, self.nlatents + self.ncauses)
        ]
        PD_to_latents = [
            (18, i) for i in range(self.ncauses, self.nlatents + self.ncauses)
        ]

        study_to_scanner = [(i, j) for i in range(9) for j in range(10, 18)]

        edges = (
            study_to_latents
            + sex_to_latents
            + scanner_to_latents
            + PD_to_latents
            + study_to_scanner
        )

        self.priors = [
            (slice(0, 9), td.OneHotCategorical(torch.tensor(P_study).to(self.device))),
            (slice(9, 10), td.Bernoulli(torch.tensor(P_sex).to(self.device))),
            (
                slice(10, 18),
                td.Normal(
                    torch.zeros(1).to(self.device), torch.ones(1).to(self.device)
                ),
            ),
            (slice(18, 19), td.Bernoulli(torch.tensor(P_PD).to(self.device))),
            (
                slice(self.ncauses, self.nlatents + self.ncauses),
                td.Normal(
                    torch.zeros(self.nlatents).to(self.device),
                    torch.ones(self.nlatents).to(self.device),
                ),
            ),
        ]

        flow_list = [
            Flow(self.nlatents + self.ncauses, edges, self.device, hm=hidden)
            for _ in range(nlayers)
        ]
        self.flow = NormalizingFlow(flow_list)

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

    @torch.no_grad()
    def sample(self, num_samples):
        z = self.priors.sample((num_samples,))
        xs, _ = self.flow.backward(z)
        return xs

    def log_likelihood(self, x):
        if type(x) is np.ndarray:
            x = torch.tensor(x.astype(np.float32))
        _, prior_logprob, log_det = self.forward(x)
        return (prior_logprob + log_det).cpu().detach().numpy()

    @torch.no_grad()
    def counterfactuals(self, x, cf_vals):
        z_obs = self(x)[0][-1]
        x_cf = x.detach().clone()
        for key in cf_vals:
            x_cf[:, key] = cf_vals[key]

        z_cf_val = self(x_cf)[0][-1]
        for key in cf_vals:
            z_obs[:, key] = z_cf_val[:, key]

        x_cf = self.backward(z_obs)[0][-1]
        return x.cpu().detach().numpy(), x_cf.cpu().detach().numpy()


class Flow(nn.Module):
    """Masked Causal Flow that uses a MADE-style network for fast-forward"""

    def __init__(self, dim, edges, device, hm=[4, 6, 4]):
        super().__init__()
        self.dim = dim
        self.net = CMADE(dim, dim * 2, edges, hm)
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
    """A sequence of Normalizing Flows is a Normalizing Flow"""

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


class MaskedLinear(nn.Linear):
    """
    Implements Masked AutoEncoder for Density Estimation, by Germain et al. 2015
    Re-implementation by Andrej Karpathy based on https://arxiv.org/abs/1502.03509
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))
        self.register_buffer("bias_mask", torch.ones(out_features))

    def set_mask(self, mask, bias_mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))
        self.bias_mask.data.copy_(torch.from_numpy(bias_mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias_mask * self.bias)


class CMADE(nn.Module):
    def __init__(self, nin, nout, edges, h_multiple):
        """
        nin: integer; number of inputs
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        edges: edges of the predefined causal graph variables
        h_multiple: list of numbers of hidden units as multiples of effect variables
        """

        super().__init__()
        self.nin = nin
        self.nout = nout
        assert self.nout % self.nin == 0, "nout must be integer multiple of nin"

        nhidden = [nin * h for h in h_multiple]

        # define a simple MLP neural net
        self.net = []
        hs = [nin] + nhidden + [nout]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend(
                [
                    MaskedLinear(h0, h1),
                    nn.ReLU(),
                ]
            )
        self.net.pop()
        self.net = nn.Sequential(*self.net)

        # Create the causal graph
        G = nx.DiGraph()
        nodes = np.arange(nin).tolist()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        # identify the parents of each node
        parents = [list(G.predecessors(n)) for n in G.nodes]

        # construct the mask matrices for hidden layers
        masks = []
        bias_masks = []
        for l in range(len(nhidden)):
            bias_mask = np.zeros(nin) > 0
            mask = np.zeros((nin, nin)) > 0
            for i in range(nin):
                mask[parents[i] + [i], i] = True

                if len(parents[i]):
                    bias_mask[i] = True

            bias_mask = np.hstack([bias_mask] * h_multiple[l])
            bias_masks.append(bias_mask)

            if l == 0:
                mask = np.hstack([mask] * h_multiple[l])
            else:
                mask = np.vstack(
                    [np.hstack([mask] * h_multiple[l])] * h_multiple[l - 1]
                )
            masks.append(mask)

        # construct the mask matrices for output layer
        k = int(nout / nin)
        mask = np.zeros((nin, nin)) > 0
        bias_mask = np.zeros(nin) > 0
        for i in range(nin):
            mask[parents[i], i] = True
            if len(parents[i]):
                bias_mask[i] = True

        mask = np.vstack([np.hstack([mask] * k)] * h_multiple[-1])
        masks.append(mask)

        bias_mask = np.hstack([np.hstack([bias_mask] * k)])
        bias_masks.append(bias_mask)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m, bm in zip(layers, masks, bias_masks):
            l.set_mask(m, bm)

    def forward(self, x):
        return self.net(x)

import torch
import numpy as np
import torch.nn as nn

from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal


class ELBO:
    def __init__(self, variational_modules, dataset_size):
        self.variational_modules = variational_modules
        self.loss_D = torch.nn.MSELoss(reduction="mean")
        self.dataset_size = dataset_size

    def loss_p(self, z, sigma_p):
        return 0.5 * z.pow(2).sum() / sigma_p**2

    def loss_q(self, sigma):
        return -sigma.sum()

    def __call__(self, pred, labels):
        loss = 0
        for module in self.variational_modules:
            # Weights loss
            loss += self.loss_p(module.z_w, module.prior_sigma_w)
            loss += self.loss_q(module.sigma_w)

            # Bias loss
            loss += self.loss_p(module.z_b, module.prior_sigma_b)
            loss += self.loss_q(module.sigma_b)

        # Negative loss likelihood
        loss += self.dataset_size * self.loss_D(pred, labels)
        return loss


class MeanFieldGaussianLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        prior_sigma_w=1.0,
        prior_sigma_b=1.0,
        init_sigma_scale=1e-7,
    ):
        super(MeanFieldGaussianLinear, self).__init__()

        # Store prior sigmas, the ELBO needs them
        self.prior_sigma_w = prior_sigma_w
        self.prior_sigma_b = prior_sigma_b

        # Weights
        self.mean_w = Parameter(torch.rand(out_features, in_features) - 0.5)
        self.sigma_w = Parameter(
            torch.log(
                init_sigma_scale * prior_sigma_w * torch.ones(out_features, in_features)
            )
        )
        self.normal_w = Normal(
            torch.zeros(out_features, in_features),
            torch.ones(out_features, in_features),
        )

        # Bias
        self.mean_b = Parameter(torch.rand(out_features) - 0.5)
        self.sigma_b = Parameter(
            torch.log(init_sigma_scale * prior_sigma_b * torch.ones(out_features))
        )
        self.normal_b = Normal(torch.zeros(out_features), torch.ones(out_features))

    def forward(self, x):
        # Sample weights
        epsilon_w = self.normal_w.sample().to(device=self.mean_w.device)
        self.z_w = self.mean_w + torch.exp(self.sigma_w) * epsilon_w

        # Sample bias
        epsilon_b = self.normal_b.sample().to(device=self.mean_b.device)
        self.z_b = self.mean_b + torch.exp(self.sigma_b) * epsilon_b

        return nn.functional.linear(x, self.z_w, bias=self.z_b)


class MeanFieldGaussianConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        prior_sigma_w=1.0,
        prior_sigma_b=1.0,
        init_sigma_scale=1e-7,
    ):
        super(MeanFieldGaussianConv2d, self).__init__()

        # Store prior sigmas, the ELBO needs them
        self.prior_sigma_w = prior_sigma_w
        self.prior_sigma_b = prior_sigma_b

        # Weights
        self.mean_w = Parameter(
            torch.rand(out_channels, in_channels, kernel_size, kernel_size) - 0.5
        )
        self.sigma_w = Parameter(
            torch.log(
                init_sigma_scale
                * prior_sigma_w
                * torch.ones(out_channels, in_channels, kernel_size, kernel_size)
            )
        )
        self.normal_w = Normal(
            torch.zeros(out_channels, in_channels, kernel_size, kernel_size),
            torch.ones(out_channels, in_channels, kernel_size, kernel_size),
        )

        # Bias
        self.mean_b = Parameter(torch.rand(out_channels) - 0.5)
        self.sigma_b = Parameter(
            torch.log(init_sigma_scale * prior_sigma_b * torch.ones(out_channels))
        )
        self.normal_b = Normal(torch.zeros(out_channels), torch.ones(out_channels))

    def forward(self, x):
        # Sample weights
        epsilon_w = self.normal_w.sample().to(device=self.mean_w.device)
        self.z_w = self.mean_w + torch.exp(self.sigma_w) * epsilon_w

        # Sample bias
        epsilon_b = self.normal_b.sample().to(device=self.mean_b.device)
        self.z_b = self.mean_b + torch.exp(self.sigma_b) * epsilon_b

        return nn.functional.conv2d(x, self.z_w, bias=self.z_b)


class VICNN(nn.Module):
    def __init__(self, prior_sigma_w=1.0, prior_sigma_b=5.0):
        super().__init__()

        self.conv1 = MeanFieldGaussianConv2d(
            1,
            16,
            kernel_size=5,
            prior_sigma_w=prior_sigma_w,
            prior_sigma_b=prior_sigma_b,
        )
        self.conv2 = MeanFieldGaussianConv2d(
            16,
            32,
            kernel_size=5,
            prior_sigma_w=prior_sigma_w,
            prior_sigma_b=prior_sigma_b,
        )
        self.linear1 = MeanFieldGaussianLinear(
            512, 128, prior_sigma_w=prior_sigma_w, prior_sigma_b=prior_sigma_b
        )
        self.linear2 = MeanFieldGaussianLinear(
            128, 10, prior_sigma_w=prior_sigma_w, prior_sigma_b=prior_sigma_b
        )

    def forward(self, x):
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = self.conv2(x)

        x = nn.functional.relu(nn.functional.max_pool2d(x, 2))
        x = x.view(-1, 512)
        x = nn.functional.relu(self.linear1(x))

        x = self.linear2(x)
        return nn.functional.log_softmax(x, dim=-1)

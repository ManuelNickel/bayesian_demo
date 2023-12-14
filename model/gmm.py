import numpy as np


class GaussianMixtureModel1D:
    def __init__(self, means=[-0.9, 0.7], sigmas=[0.2, 0.6]):
        assert len(means) == len(sigmas)
        self.means = means
        self.sigmas = sigmas
        self.z = 0
        for mean, sigma in zip(self.means, self.sigmas):
            self.z += self.normalization(sigma)

    def gaussian(self, x, mean, sigma):
        return np.exp(-0.5 * (x - mean) ** 2 / sigma**2)

    def grad(self, x, mean, sigma):
        return self.gaussian(x, mean, sigma) * ((mean - x) / sigma**2)

    def normalization(self, sigma):
        return (2 * np.pi) ** 0.5 * sigma

    def __call__(self, x):
        y = 0
        for mean, sigma in zip(self.means, self.sigmas):
            y += self.gaussian(x, mean, sigma)
        return y / self.z

    def gradient(self, x):
        y = 0
        for mean, sigma in zip(self.means, self.sigmas):
            y += self.grad(x, mean, sigma)
        return y / self.z

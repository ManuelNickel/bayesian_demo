# The Bayesian Demo Notebooks
This little repository serves as a demo playground for Bayesian Neural Networks. Algorithms for some basic Bayesian methods in the context of machine learning are implemented with the corresponding theory side-by-side. The methods are then demonstrated on some toy datasets.

## Sampling-based methods
Two classical sampling-based methods are demonstrated on very simple one-dimensional toy distributions. These methods are guaranteed to exactly represent a distribution from which they sample. However, they are generally too expensive for large contemporary neural networks.
- [Random Walk Metropolis Hastings](demo-books/rwmh.ipynb): Simplest method to sample from a posterior distribution without knowing the evidence.
- [Hamiltonian Monte Carlo](demo-books/hmc.ipynb): A much more efficient sampling method based on Hamiltonian dynamics.

## Approximate methods for larger models
Some established methods for approximating a posterior distribution are demonstrated here. These methods scale to larger neural networks.
A script based on [maximum likelihood estimation](demo-books/mlecnn.ipynb) implements a base setup and serves as a tool for direct comparison to the Bayesian designs.
- [Stochastic Variational Inference](demo-books/svicnn.ipynb): Approximate the posterior distribution by learning a chosen family of distributions.
- [Stochastic Variational Inference with Parametrizable Priors](demo-books/svipp.ipynb): In addition to the above setup, also learn prior distributions over network parameters.
- [Laplace Approximation](demo-books/lapapp.ipynb): Approximate the posterior distribution by computing its Laplace approximation.

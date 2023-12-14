import torch
import torchvision

from torch.utils.data.sampler import SubsetRandomSampler


def make_mnist(filteredClass=None, removeFiltered=True):
    train = torchvision.datasets.MNIST(
        "./data/",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

    test = torchvision.datasets.MNIST(
        "./data/",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

    if filteredClass is not None:
        train_loader = torch.utils.data.DataLoader(train, batch_size=len(train))
        train_labels = next(iter(train_loader))[1].squeeze()

        test_loader = torch.utils.data.DataLoader(test, batch_size=len(test))
        test_labels = next(iter(test_loader))[1].squeeze()

        if removeFiltered:
            trainIndices = torch.nonzero(train_labels != filteredClass).squeeze()
            testIndices = torch.nonzero(test_labels != filteredClass).squeeze()
        else:
            trainIndices = torch.nonzero(train_labels == filteredClass).squeeze()
            testIndices = torch.nonzero(test_labels == filteredClass).squeeze()

        train = torch.utils.data.Subset(train, trainIndices)
        test = torch.utils.data.Subset(test, testIndices)

    return train, test

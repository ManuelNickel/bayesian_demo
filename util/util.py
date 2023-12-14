import scipy
import torch
import numpy as np
import matplotlib.pyplot as plt


def kl_divergence(q, p, eps=1e-15):
    assert np.all(
        q >= 0
    ), f"Expected valid probability distribution for q but encountered negative values"
    assert np.all(
        p >= 0
    ), f"Expected valid probability distribution for p but encountered negative values"

    # Let's add a tiny value to p so that KL is not always infinite
    p = p + eps
    p = p / np.sum(p)
    q = q / np.sum(q)
    kl = scipy.special.rel_entr(q, p)
    return np.sum(kl)


def trace(model, samples, x_min=-2.5, x_max=2.5, bins=100, kl_skip=50):
    # Plot samples against their time step
    n = len(samples)
    t = np.arange(n)
    fig, ax = plt.subplots()
    ax.plot(t, samples, c="k", linewidth=0.3)

    # Compute acceptance rate
    rejected = np.isclose(np.diff(samples), 0)
    acceptance_rate = (n - np.count_nonzero(rejected)) / n

    # Plot target and final sampled distribution vertically on the far left as reference
    bin_edges = np.arange(x_min, x_max, (x_max - x_min) / bins)
    p, _ = np.histogram(samples, bins=bin_edges, density=True)
    # Euler approximation to find value centered in each bin
    x = bin_edges[:-1] + 0.5 * np.diff(bin_edges)
    q = model(x)
    y_max = max(max(q), max(p))
    yp = p / y_max * 0.1 * n
    yq = q / y_max * 0.1 * n
    ax.plot(yq, x, c="r")
    ax.plot(yp, x, c="b")

    # Compute KL divergence
    kl = kl_divergence(q, p)

    # Compute KL divergence over time
    kls = np.empty(n - kl_skip)
    for i in range(0, n - kl_skip):
        p, _ = np.histogram(samples[: kl_skip + i], bins=bin_edges, density=True)
        kls[i] = kl_divergence(q, p)

    # Plot KL divergence wrt the right axis
    ax_right = ax.twinx()
    x = np.arange(kl_skip, n)
    ax_right.plot(x, kls, c="g")

    plt.title(
        f"Sample trace - {acceptance_rate * 100:.2f}% acceptance, {kl:.2f} KL divergence"
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Sample value", c="r")
    ax.tick_params(axis="y", labelcolor="r")
    ax_right.set_ylabel("KL divergence", c="g")
    ax_right.tick_params(axis="y", labelcolor="g")
    ax_right.set_ylim([0, 15])
    plt.show()


def plot_distribution(model, samples, x_min=-2.5, x_max=2.5, bins=100):
    x = np.arange(x_min, x_max, 0.001)
    y = model(x)

    plt.hist(samples, bins=bins, density=True, label="Sample histogram")
    plt.plot(x, y, c="r", label="target PDF")
    plt.legend(loc="upper right")

    plt.title("Target distribution vs. sample histogram")
    plt.xlabel("x value")
    plt.ylabel("PDF and histogram")
    plt.show()


def show_images(images, samples, labels):
    fig, axes = plt.subplots(len(images), 2, figsize=(10, 22))
    for image, sample, label, ax in zip(images, samples, labels, axes):
        ax[0].imshow(image)
        ax[0].set_ylabel(label.item())
        ax[0].xaxis.set_tick_params(labelbottom=False)
        ax[0].yaxis.set_tick_params(labelleft=False)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_ylim([-0.2, 1.2])
        ax[1].boxplot(sample.transpose(1, 0))
        ax[1].set_xticklabels([i for i in range(sample.shape[1])])


def run_model_on_data(
    model,
    dataset,
    num_images=None,
    num_samples_per_image=128,
    select_class=None,
    shuffle=True,
):
    if select_class is not None:
        indices = np.where(dataset.targets == select_class)[0]
        dataset = torch.utils.data.Subset(dataset, indices)

    if num_images is None:
        num_images = len(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=num_images, shuffle=shuffle
    )
    samples = torch.zeros((num_samples_per_image, num_images, 10))
    images, labels = next(iter(data_loader))

    with torch.no_grad():
        for i in range(num_samples_per_image):
            samples[i, :, :] = torch.exp(model(images))

    images = images.squeeze()
    samples = samples.transpose(1, 0)
    return images, labels, samples


def store_for_later():
    # Pick n test images and sample each num_samples times
    # To select only a certain class, assign filter_class the corresponding number
    num_samples_per_image = 256
    select_class = 5

    _, test = make_mnist()
    images, labels, samples = run_model_on_data(
        model, test, None, num_samples_per_image, select_class
    )

    median = np.median(samples, axis=1)
    upper_percentile = np.percentile(samples, 90, axis=1)
    proposals = upper_percentile > 0.05
    indices = np.where(proposals.sum(axis=1) == 1)[0]
    accepted = np.where(median[indices] > 0.9)[0]
    print(sum(accepted))

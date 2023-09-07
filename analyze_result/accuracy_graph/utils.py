import math

accuracy_bases = {
    "MNIST": 99.36,
    "EMNIST": 92.62,
    "GTSRB": 93.24,
    "CIFAR10": 89.12,
}


def sig(x):
    return 1 / (1 + math.exp(-x))


def get_neuron_point(dataset, separating_accuracy, removing_accuracy):
    accuracy = accuracy_bases[dataset]

    point = -sig(accuracy - separating_accuracy) + sig(
        (separating_accuracy - removing_accuracy) / 5
    )

    return point

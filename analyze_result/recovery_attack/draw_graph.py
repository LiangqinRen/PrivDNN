import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from scipy import linalg, optimize
from scipy.interpolate import interp1d


sys.path.append("../cipher_operations/")

import CIFAR10


def draw_recovery_attack_graph(
    percents,
    separate_accuracies,
    remove_accuracies,
    recover_accuracies,
    cifar10_cipher_neurons,
):
    plt.figure(figsize=(7, 4))

    plt.xlabel("Selected Neurons (%)")
    plt.ylabel("%")

    plt.xticks(np.linspace(50, 100, 11))  # [50, 105, 5]
    plt.yticks(np.linspace(10, 100, 10))  # [10, 110, 10]

    percents_inter = np.linspace(50, 100, 101)

    separate_inter = interp1d(percents, separate_accuracies, kind="linear")
    separate_accuracies_percents_inter = separate_inter(percents_inter)
    plt.plot(
        percents_inter,
        separate_accuracies_percents_inter,
        linestyle="solid",
        color="green",
        label="$A_s$",
    )

    remove_inter = interp1d(percents, remove_accuracies, kind="linear")
    remove_accuracies_percents_inter = remove_inter(percents_inter)
    plt.plot(
        percents_inter,
        remove_accuracies_percents_inter,
        linestyle="solid",
        color="orange",
        label="$A_r$",
    )

    recover_inter = interp1d(percents, recover_accuracies, kind="linear")
    recover_accuracies_percents_inter = recover_inter(percents_inter)
    plt.plot(
        percents_inter,
        recover_accuracies_percents_inter,
        linestyle="solid",
        color="brown",
        label="$A_{rec}$",
    )

    cipher_neurons_inter = interp1d(percents, cifar10_cipher_neurons, kind="linear")
    cipher_neurons_accuracies_percents_inter = cipher_neurons_inter(percents_inter)
    plt.plot(
        percents_inter,
        cipher_neurons_accuracies_percents_inter,
        linestyle="solid",
        color="purple",
        label="$RE_{n}$",
    )

    plt.scatter(percents[5], separate_accuracies[5], color="red", zorder=2)
    plt.annotate(
        f"{separate_accuracies[5]:.2f}",
        xy=(percents[5], separate_accuracies[5]),
        xytext=(percents[5] + 1, separate_accuracies[5] + 1),
        color="red",
    )
    plt.scatter(percents[5], remove_accuracies[5], color="red", zorder=2)
    plt.annotate(
        f"{remove_accuracies[5]:.2f}",
        xy=(percents[5], remove_accuracies[5]),
        xytext=(percents[5] - 1, remove_accuracies[5] - 7),
        color="red",
    )
    plt.scatter(percents[5], recover_accuracies[5], color="red", zorder=2)
    plt.annotate(
        f"{recover_accuracies[5]:.2f}",
        xy=(percents[5], recover_accuracies[5]),
        xytext=(percents[5] + 1, recover_accuracies[5] + 1),
        color="red",
    )
    plt.scatter(percents[5], cifar10_cipher_neurons[5], color="red", zorder=2)
    plt.annotate(
        f"{cifar10_cipher_neurons[5]:.2f}",
        xy=(percents[5], cifar10_cipher_neurons[5]),
        xytext=(percents[5] + 1, cifar10_cipher_neurons[5] + 1),
        color="red",
    )

    plt.legend(loc="best")

    for i in range(10, 110, 10):
        plt.axhline(i, linestyle="--", color="gray", alpha=0.3)

    plt.savefig(f"recovery_attack.png", dpi=1024)
    plt.close()


if __name__ == "__main__":
    percents = [*range(50, 105, 5)]
    separate_accuracies = [
        89.48,
        89.38,
        89.40,
        89.40,
        89.38,
        89.44,
        89.52,
        89.42,
        89.40,
        89.28,
        89.12,
    ]
    remove_accuracies = [
        53.88,
        49.62,
        43.92,
        34.88,
        29.08,
        19.90,
        16.90,
        12.08,
        9.84,
        9.40,
        9.78,
    ]
    recover_accuracies = [
        82.84,
        82.02,
        80.96,
        79.14,
        79.04,
        27.72,
        27.60,
        26.18,
        24.56,
        23.12,
        22.20,
    ]
    cifar10_cipher_neurons = []
    for i in percents:
        operations = CIFAR10.count_neurons(i)
        cifar10_cipher_neurons.append(operations[0] / operations[1] * 100)

    draw_recovery_attack_graph(
        percents,
        separate_accuracies,
        remove_accuracies,
        recover_accuracies,
        cifar10_cipher_neurons,
    )

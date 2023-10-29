import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import GTSRB
import CIFAR10


def count_operations(percent_list):
    gtsrb_operations = {"+": [], "*": []}
    cifar10_operations = {"+": [], "*": []}
    for percent in percent_list:
        gtsrb_operations["+"].append(GTSRB.count_cipher_addition(percent))
        gtsrb_operations["*"].append(GTSRB.count_cipher_multiplication(percent))

        cifar10_operations["+"].append(CIFAR10.count_cipher_addition(percent))
        cifar10_operations["*"].append(CIFAR10.count_cipher_multiplication(percent))

    return gtsrb_operations, cifar10_operations


def draw_operations_graph(x, y_gtsrb, y_cifar10):
    plt.xlabel("Selected Neurons (%)")
    plt.ylabel("Cipher Operations (%)")

    y_gtsrb_add = [iter[0] / iter[1] * 100 for iter in y_gtsrb["+"]]
    y_gtsrb_mul = [iter[0] / iter[1] * 100 for iter in y_gtsrb["*"]]
    plt.plot(x, y_gtsrb_add, linestyle="solid", color="blue", label="GTSRB +")
    plt.plot(x, y_gtsrb_mul, linestyle="dotted", color="red", label="GTSRB *")

    y_cifar10_add = [iter[0] / iter[1] * 100 for iter in y_cifar10["+"]]
    y_cifar10_mul = [iter[0] / iter[1] * 100 for iter in y_cifar10["*"]]
    plt.plot(x, y_cifar10_add, linestyle="solid", color="blue", label="CIFAR-10 +")
    plt.plot(x, y_cifar10_mul, linestyle="dotted", color="red", label="CIFAR-10 *")

    plt.legend(loc="upper left")

    plt.savefig(f"cipher_operations.png")
    plt.close()


if __name__ == "__main__":
    percents = [*range(50, 105, 5)]
    gtsrb_data, cifar10_data = count_operations(percents)
    draw_operations_graph(percents, gtsrb_data, cifar10_data)

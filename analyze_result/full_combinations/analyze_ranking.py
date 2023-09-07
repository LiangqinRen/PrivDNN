import utils

import random
import numpy as np

file = "EMNIST.npy"

MNIST_pruning_norm_selections = [
    {1: [0], 2: [3]},
    {1: [0], 2: [3, 14]},
    {1: [0], 2: [3, 13, 14]},
    {1: [0], 2: [3, 5, 13, 14]},
    {1: [0], 2: [1, 3, 5, 13, 14]},
    {1: [0, 5], 2: [3, 14]},
    {1: [0, 5], 2: [3, 13, 14]},
    {1: [0, 5], 2: [3, 5, 13, 14]},
    {1: [0, 5], 2: [1, 3, 5, 13, 14]},
    {1: [0, 5], 2: [1, 3, 5, 6, 13, 14]},
]

MNIST_pruning_fpgm_selections = [
    {1: [0], 2: [1]},
    {1: [0], 2: [1, 14]},
    {1: [0], 2: [1, 3, 14]},
    {1: [0], 2: [1, 3, 12, 14]},
    {1: [0], 2: [1, 3, 12, 14, 15]},
    {1: [0, 4], 2: [1, 14]},
    {1: [0, 4], 2: [1, 3, 14]},
    {1: [0, 4], 2: [1, 3, 12, 14]},
    {1: [0, 4], 2: [1, 3, 12, 14, 15]},
    {1: [0, 4], 2: [1, 3, 12, 13, 14, 15]},
]

MNIST_pruning_hrank_selections = [
    {1: [4], 2: [13]},
    {1: [4], 2: [6, 13]},
    {1: [4], 2: [5, 6, 13]},
    {1: [4], 2: [4, 5, 6, 13]},
    {1: [4], 2: [1, 4, 5, 6, 13]},
    {1: [0, 4], 2: [6, 13]},
    {1: [0, 4], 2: [5, 6, 13]},
    {1: [0, 4], 2: [4, 5, 6, 13]},
    {1: [0, 4], 2: [1, 4, 5, 6, 13]},
    {1: [0, 4], 2: [1, 4, 5, 6, 13, 14]},
]

MNIST_pruning_greedy_selections = [
    {1: [1], 2: [1]},
    {1: [1], 2: [1, 2]},
    {1: [1], 2: [1, 2, 15]},
    {1: [1], 2: [1, 2, 3, 15]},
    {1: [1], 2: [1, 2, 3, 10, 15]},
    {1: [1, 3], 2: [1, 2]},
    {1: [1, 3], 2: [1, 2, 15]},
    {1: [1, 3], 2: [1, 2, 3, 15]},
    {1: [1, 3], 2: [1, 2, 3, 10, 15]},
    {1: [1, 3], 2: [1, 2, 3, 10, 11, 15]},
]

MNIST_greedy_selections = [
    {1: [4], 2: [8]},
    {1: [4], 2: [8, 11]},
    {1: [4], 2: [8, 11, 0]},
    {1: [4], 2: [8, 11, 0, 9]},
    {1: [4], 2: [8, 11, 0, 9, 4]},
    {1: [4, 5], 2: [8, 11]},
    {1: [4, 5], 2: [8, 11, 1]},
    {1: [4, 5], 2: [8, 11, 1, 6]},
    {1: [4, 5], 2: [8, 11, 1, 6, 3]},
    {1: [4, 5], 2: [8, 11, 1, 6, 3, 2]},
]

MNIST_pfec_and_greedy_selections = [
    {1: [0], 2: [14]},
    {1: [0], 2: [14, 5]},
    {1: [0], 2: [14, 5, 1]},
    {1: [0], 2: [14, 12, 1, 6]},
    {1: [0], 2: [14, 12, 1, 15, 6]},
    {1: [4, 5], 2: [13, 14]},
    {1: [4, 5], 2: [1, 14, 13]},
    {1: [4, 5], 2: [12, 1, 14, 6]},
    {1: [4, 5], 2: [12, 1, 11, 6, 15]},
    {1: [4, 5], 2: [12, 1, 11, 6, 2, 15]},
]

MNIST_fpgm_and_greedy_selections = [
    {1: [0], 2: [14]},
    {1: [0], 2: [14, 12]},
    {1: [0], 2: [14, 12, 1]},
    {1: [0], 2: [14, 12, 1, 15]},
    {1: [0], 2: [14, 12, 1, 15, 6]},
    {1: [4, 5], 2: [12, 1]},
    {1: [4, 5], 2: [12, 1, 14]},
    {1: [4, 5], 2: [12, 1, 14, 15]},
    {1: [4, 5], 2: [12, 1, 11, 6, 15]},
    {1: [4, 5], 2: [12, 1, 11, 6, 2, 15]},
]

EMNIST_pruning_norm_selections = [
    {1: [8], 2: [3]},
    {1: [8], 2: [3, 14]},
    {1: [8], 2: [3, 14, 18]},
    {1: [5, 8], 2: [3, 14]},
    {1: [5, 8], 2: [3, 14, 18]},
    {1: [5, 8], 2: [3, 14, 16, 18]},
]

EMNIST_pruning_fpgm_selections = [
    {1: [8], 2: [3]},
    {1: [8], 2: [3, 14]},
    {1: [8], 2: [3, 14, 16]},
    {1: [8, 5], 2: [3, 14]},
    {1: [8, 5], 2: [3, 14, 16]},
    {1: [8, 5], 2: [3, 14, 16, 4]},
]

EMNIST_pruning_hrank_selections = [
    {1: [3], 2: [3]},
    {1: [3], 2: [3, 16]},
    {1: [3], 2: [3, 16, 12]},
    {1: [3, 9], 2: [3, 16]},
    {1: [3, 9], 2: [3, 16, 12]},
    {1: [3, 9], 2: [3, 16, 12, 14]},
]

EMNIST_pruning_greedy_selections = [
    {1: [2], 2: [13]},
    {1: [2], 2: [13, 19]},
    {1: [2], 2: [13, 19, 11]},
    {1: [2, 4], 2: [13, 19]},
    {1: [2, 4], 2: [13, 19, 11]},
    {1: [2, 4], 2: [13, 19, 11, 17]},
]

EMNIST_greedy_selections = [
    {1: [3], 2: [3]},
    {1: [3], 2: [3, 8]},
    {1: [3], 2: [3, 8, 11]},
    {1: [3, 9], 2: [3, 8]},
    {1: [3, 9], 2: [3, 8, 12]},
    {1: [3, 9], 2: [3, 8, 12, 6]},
]

EMNIST_pfec_and_greedy_selections = [
    {1: [5], 2: [3]},
    {1: [5], 2: [3, 14]},
    {1: [5], 2: [3, 14, 18]},
    {1: [9, 2], 2: [3, 14]},
    {1: [9, 2], 2: [3, 0, 18]},
    {1: [9, 2], 2: [3, 0, 18, 14]},
]

EMNIST_fpgm_and_greedy_selections = [
    {1: [5], 2: [3]},
    {1: [5], 2: [3, 4]},
    {1: [5], 2: [3, 4, 14]},
    {1: [9, 2], 2: [3, 4]},
    {1: [9, 2], 2: [3, 4, 14]},
    {1: [9, 2], 2: [3, 0, 18, 14]},
]

selections = (
    [
        # MNIST_pruning_norm_selections,
        # MNIST_pruning_fpgm_selections,
        # MNIST_pruning_hrank_selections,
        # MNIST_pruning_greedy_selections,
        MNIST_greedy_selections,
        MNIST_pfec_and_greedy_selections,
        MNIST_fpgm_and_greedy_selections,
    ]
    if file == "MNIST.npy"
    else [
        # EMNIST_pruning_norm_selections,
        # EMNIST_pruning_fpgm_selections,
        # EMNIST_pruning_hrank_selections,
        # EMNIST_pruning_greedy_selections,
        EMNIST_greedy_selections,
        EMNIST_pfec_and_greedy_selections,
        EMNIST_fpgm_and_greedy_selections,
    ]
)

# selections need to be sorted
results = np.load(file, allow_pickle=True)
results = results.tolist()
first_layer_count = 2
add_factor = 5 if file == "MNIST.npy" else 3
complete_selections = {}
for first_count in range(1, first_layer_count + 1):
    for second_count in range(first_count, first_count + add_factor):
        complete_selections[(first_count, second_count)] = []

for combination in results:
    selection = combination[1]
    if (len(selection[1]), len(selection[2])) == (1, 6):
        continue

    separate_accuracy = combination[2]
    remove_accuracy = combination[3]
    point = utils.get_neuron_point(file, separate_accuracy, remove_accuracy, False)

    complete_selections[(len(selection[1]), len(selection[2]))].append(
        (point, selection, separate_accuracy, remove_accuracy)
    )

for count in complete_selections:
    data = complete_selections[count]
    data.sort(key=lambda point: point[0], reverse=True)
    complete_selections[count] = data


i = 0
for first_count in range(1, first_layer_count + 1):
    for second_count in range(first_count, first_count + add_factor):
        data = complete_selections[(first_count, second_count)]
        ranking = []
        for selection in selections:
            selection[i][1] = sorted(selection[i][1])
            selection[i][2] = sorted(selection[i][2])

            for j, content in enumerate(data, 1):
                if (
                    tuple(selection[i][1]) == content[1][1]
                    and tuple(selection[i][2]) == content[1][2]
                ):
                    ranking.append(j)
        print(
            f"In {len(data)} selections of [{first_count}, {second_count}], five rankings are {ranking}"
        )
        i += 1

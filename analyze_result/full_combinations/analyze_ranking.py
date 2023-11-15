import utils

import numpy as np

file = "MNIST"

# copy selections to here manually
MNIST_pruning_norm_selections = [
    {1: [0], 2: [3]},
    {1: [0], 2: [3, 14]},
    {1: [0], 2: [3, 13, 14]},
    {1: [0], 2: [3, 5, 13, 14]},
    {1: [0], 2: [1, 3, 5, 13, 14]},
    {1: [0], 2: [1, 3, 5, 6, 13, 14]},
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
    {1: [0], 2: [1, 3, 12, 13, 14, 15]},
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
    {1: [4], 2: [1, 4, 5, 6, 13, 14]},
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
    {1: [1], 2: [1, 2, 3, 10, 11, 15]},
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
    {1: [4], 2: [8, 11, 0, 9, 4, 14]},
    {1: [4, 5], 2: [8, 11]},
    {1: [4, 5], 2: [8, 11, 1]},
    {1: [4, 5], 2: [8, 11, 1, 6]},
    {1: [4, 5], 2: [8, 11, 1, 6, 3]},
    {1: [4, 5], 2: [8, 11, 1, 6, 3, 2]},
]

MNIST_pfec_and_greedy_selections = [
    {1: [4], 2: [14]},
    {1: [4], 2: [14, 13]},
    {1: [4], 2: [1, 14, 13]},
    {1: [4], 2: [1, 14, 12, 13]},
    {1: [4], 2: [11, 15, 6, 12, 1]},
    {1: [4], 2: [11, 15, 6, 0, 2, 12]},
    {1: [4, 5], 2: [13, 14]},
    {1: [4, 5], 2: [1, 14, 13]},
    {1: [4, 5], 2: [12, 1, 14, 6]},
    {1: [4, 5], 2: [12, 1, 11, 6, 15]},
    {1: [4, 5], 2: [12, 1, 11, 6, 2, 15]},
]

MNIST_fpgm_and_greedy_selections = [
    {1: [4], 2: [1]},
    {1: [4], 2: [1, 14]},
    {1: [4], 2: [1, 14, 12]},
    {1: [4], 2: [1, 14, 12, 13]},
    {1: [4], 2: [11, 15, 6, 12, 1]},
    {1: [4], 2: [11, 15, 6, 0, 2, 12]},
    {1: [4, 5], 2: [12, 1]},
    {1: [4, 5], 2: [12, 1, 14]},
    {1: [4, 5], 2: [12, 1, 14, 15]},
    {1: [4, 5], 2: [12, 1, 11, 6, 15]},
    {1: [4, 5], 2: [12, 1, 11, 6, 2, 15]},
]

EMNIST_pruning_norm_selections = [
    {1: [9], 2: [15]},
    {1: [9], 2: [6, 15]},
    {1: [9], 2: [6, 7, 15]},
    {1: [9], 2: [6, 7, 12, 15]},
    {1: [3, 9], 2: [6, 15]},
    {1: [3, 9], 2: [6, 7, 15]},
    {1: [3, 9], 2: [6, 7, 12, 15]},
]

EMNIST_pruning_fpgm_selections = [
    {1: [3], 2: [15]},
    {1: [3], 2: [3, 15]},
    {1: [3], 2: [3, 7, 15]},
    {1: [3], 2: [3, 7, 8, 15]},
    {1: [3, 9], 2: [3, 15]},
    {1: [3, 9], 2: [3, 15, 17]},
    {1: [3, 9], 2: [3, 7, 8, 15]},
]

EMNIST_pruning_hrank_selections = [
    {1: [2], 2: [7]},
    {1: [2], 2: [5, 7]},
    {1: [2], 2: [5, 7, 19]},
    {1: [2], 2: [5, 7, 15, 19]},
    {1: [2, 3], 2: [5, 7]},
    {1: [2, 3], 2: [5, 7, 19]},
    {1: [2, 3], 2: [5, 7, 15, 19]},
]

EMNIST_pruning_greedy_selections = [
    {1: [4], 2: [19]},
    {1: [4], 2: [7, 19]},
    {1: [4], 2: [7, 8, 19]},
    {1: [4], 2: [7, 8, 15, 19]},
    {1: [4, 8], 2: [7, 19]},
    {1: [4, 8], 2: [7, 8, 19]},
    {1: [4, 8], 2: [7, 8, 15, 19]},
]

EMNIST_greedy_selections = [
    {1: [4], 2: [12]},
    {1: [4], 2: [12, 8]},
    {1: [4], 2: [12, 8, 2]},
    {1: [4], 2: [12, 8, 2, 11]},
    {1: [4, 2], 2: [8, 12]},
    {1: [4, 2], 2: [8, 12, 2]},
    {1: [4, 2], 2: [8, 12, 2, 7]},
]

EMNIST_pfec_and_greedy_selections = [
    {1: [4], 2: [6]},
    {1: [4], 2: [12, 7]},
    {1: [4], 2: [12, 7, 0]},
    {1: [4], 2: [12, 7, 0, 6]},
    {1: [4, 2], 2: [12, 7]},
    {1: [4, 2], 2: [12, 7, 0]},
    {1: [4, 2], 2: [12, 7, 0, 6]},
]

EMNIST_fpgm_and_greedy_selections = [
    {1: [4], 2: [3]},
    {1: [4], 2: [8, 7]},
    {1: [4], 2: [12, 8, 7]},
    {1: [4], 2: [12, 8, 2, 6]},
    {1: [4, 2], 2: [8, 7]},
    {1: [4, 2], 2: [8, 12, 7]},
    {1: [4, 2], 2: [8, 12, 2, 7]},
]

our_selections = (
    [
        MNIST_pruning_norm_selections,
        MNIST_pruning_fpgm_selections,
        MNIST_pruning_hrank_selections,
        MNIST_pruning_greedy_selections,
        MNIST_greedy_selections,
        MNIST_pfec_and_greedy_selections,
        MNIST_fpgm_and_greedy_selections,
    ]
    if file == "MNIST"
    else [
        EMNIST_pruning_norm_selections,
        EMNIST_pruning_fpgm_selections,
        EMNIST_pruning_hrank_selections,
        EMNIST_pruning_greedy_selections,
        EMNIST_greedy_selections,
        EMNIST_pfec_and_greedy_selections,
        EMNIST_fpgm_and_greedy_selections,
    ]
)

# construct all possible combinations
second_layer_limit = 6 if file == "MNIST" else 4
complete_selections = {}
for first_count in range(1, 3):
    for second_count in range(first_count, second_layer_limit + 1):
        complete_selections[(first_count, second_count)] = []

# load all possible combinations from the file
results = np.load(f"{file}.npy", allow_pickle=True)
for combination in results.tolist():
    selection = combination[1]
    separate_accuracy = combination[2]
    remove_accuracy = combination[3]
    point = utils.get_neuron_point(file, separate_accuracy, remove_accuracy)

    complete_selections[(len(selection[1]), len(selection[2]))].append(
        (point, selection, separate_accuracy, remove_accuracy)
    )

# sort selections by the point
for selections in complete_selections:
    original_selections = complete_selections[selections]
    original_selections.sort(key=lambda point: point[0], reverse=True)
    complete_selections[selections] = original_selections

# sort our selections neurons order
for algorithm_selection in our_selections:
    for selection in algorithm_selection:
        selection[1].sort()
        selection[2].sort()

# search for rankings
for first_count in range(1, 3):
    for second_count in range(first_count, second_layer_limit + 1):
        order_selections = complete_selections[(first_count, second_count)]

        # filter our selections with current neurons count
        iter_our_selection = []
        for our_selection in our_selections:
            for selection in our_selection:
                if (
                    len(selection[1]) == first_count
                    and len(selection[2]) == second_count
                ):
                    iter_our_selection.append(selection)

        ranking = []
        for selection in iter_our_selection:
            for i, data in enumerate(order_selections, 1):
                if (
                    tuple(selection[1]) == data[1][1]
                    and tuple(selection[2]) == data[1][2]
                ):
                    ranking.append(i)

        print(
            f"In {len(order_selections)} selections of [{first_count}, {second_count}], {len(iter_our_selection)} rankings are {ranking}"
        )

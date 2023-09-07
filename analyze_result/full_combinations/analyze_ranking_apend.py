import utils

import random
import numpy as np

file = "EMNIST"

MNIST = [
    {1: [0], 2: [1, 3, 5, 6, 13, 14]},
    {1: [0], 2: [1, 14, 3, 12, 15, 13]},
    {1: [4], 2: [13, 6, 5, 4, 1, 14]},
    {1: [1], 2: [1, 2, 15, 3, 10, 11]},
    {1: [0], 2: [14, 12, 1, 15, 6, 2]},
    {1: [0], 2: [14, 12, 1, 15, 6, 2]},
]
EMNIST = [
    {1: [8], 2: [3, 14, 16, 18]},
    {1: [8], 2: [3, 14, 16, 4]},
    {1: [3], 2: [3, 16, 12, 14]},
    {1: [2], 2: [13, 19, 11, 17]},
    {1: [5], 2: [3, 4, 14, 18]},
    {1: [5], 2: [3, 4, 14, 18]},
]

results = np.load(f"{file}_1_4.npy", allow_pickle=True)
results = results.tolist()

complete_selections = []
for combination in results:
    selection = combination[1]

    if (len(selection[1]), len(selection[2])) == (1, 4):
        separate_accuracy = combination[2]
        remove_accuracy = combination[3]
        point = utils.get_neuron_point(
            file,
            separate_accuracy,
            remove_accuracy,
        )

        complete_selections.append(
            (point, selection, separate_accuracy, remove_accuracy)
        )


complete_selections.sort(key=lambda point: point[0], reverse=True)

ranking = []
for selection in EMNIST:
    selection[1] = sorted(selection[1])
    selection[2] = sorted(selection[2])
    for i, iter in enumerate(complete_selections):
        if list(selection[1]) == list(iter[1][1]) and list(selection[2]) == list(
            iter[1][2]
        ):
            ranking.append(i)
            break

print(ranking)

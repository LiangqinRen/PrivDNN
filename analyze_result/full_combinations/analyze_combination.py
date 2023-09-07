import utils
import numpy as np

file = "EMNIST_1_4"

results = np.load(f"{file}.npy", allow_pickle=True)
results = results.tolist()

print(f"There are {len(results)} possible combinations")

neuron_combination = {}
for line in results:
    key = str([len(line[1][1]), len(line[1][2])])
    if key not in neuron_combination:
        neuron_combination[key] = []

    neuron_combination[key].append(
        [
            line[1],
            line[2],
            line[3],
        ]
    )


top = 1
for neurons in neuron_combination.keys():
    points = []  # float, [], {}
    for combination in neuron_combination[neurons]:
        selected_neurons = combination[0]
        separating_accuracy = float(combination[1])
        removing_accuracy = float(combination[2])

        points.append(
            [
                utils.get_neuron_point(
                    "EMNIST", separating_accuracy, removing_accuracy
                ),
                [separating_accuracy, removing_accuracy],
                selected_neurons,
            ]
        )

    points.sort(key=lambda point: point[0], reverse=True)

    print(
        f"top {top} best selections of {neurons} in {len(neuron_combination[neurons])} combinations"
    )
    for index, point in enumerate(points[:top]):
        print(
            f"{index}|{float(point[0]):.2f}, {point[2]}, [{point[1][0]:.2f}, {point[1][1]:.2f}]"
        )

    print(
        f"top {top} worst selections of {neurons} in {len(neuron_combination[neurons])} combinations"
    )
    for index, point in enumerate(points[-top:]):
        print(
            f"{index}|{float(point[0]):.2f}, {point[2]}, [{point[1][0]:.2f}, {point[1][1]:.2f}]"
        )

    print()

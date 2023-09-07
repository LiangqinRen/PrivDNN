import utils

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def calculate_selection_ranking(dataset, data_path):
    npy_data = np.load(data_path, allow_pickle=True)
    npy_data = npy_data.tolist()

    valid_selections = {
        "MNIST": [
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
        ],
        "EMNIST": [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (2, 4)],
    }

    selection_data = []
    for data in npy_data:
        if (len(data[1][1]), len(data[1][2])) in valid_selections[dataset]:
            selection_data.append(data)

    ranking_data = []
    for data in selection_data:
        # data format: [index, selection, A_s, A_r]
        ranking_data.append(
            (
                utils.get_neuron_point(dataset, data[2], data[3]),
                data[1],
                (data[2], data[3]),
            )
        )

    ranking_data = sorted(ranking_data, key=lambda data: data[0], reverse=True)
    return ranking_data


def extract_data(ranking_data, data_categoty):
    filtered_data = []
    for data in ranking_data:
        if len(data[1][1]) == data_categoty[0] and len(data[1][2]) == data_categoty[1]:
            filtered_data.append(data)

    return filtered_data


def draw_ranking_accuracy(point_count, dataset, data_source):
    separate_accuracy = []
    remove_accuracy = []
    ranking = []
    for i, data in enumerate(data_source):
        if i % int(len(data_source) / point_count) == 0:
            ranking.append(i)
            separate_accuracy.append(data[2][0])
            remove_accuracy.append(data[2][1])

    plt.xlabel("Ranking")
    plt.ylabel("A$_s$/A$_r$")

    plt.plot(ranking, separate_accuracy, linestyle="solid", color="blue", label="A$_s$")
    plt.plot(ranking, remove_accuracy, linestyle="solid", color="black", label="A$_r$")

    plt.legend(loc="lower left")
    plt.title(dataset)

    plt.savefig(f"{dataset}_accuracy.png")
    plt.close()


def draw_ranking_accuracy_graph(MNIST_data_source, EMNIST_data_source):
    point_count = 100

    draw_ranking_accuracy(point_count, "MNIST", MNIST_data_source)
    draw_ranking_accuracy(point_count, "EMNIST", EMNIST_data_source)


def draw_accuracy_distribution(dataset, data_source):
    separate_accuracy = []
    remove_accuracy = []
    for _, data in enumerate(data_source):
        separate_accuracy.append(data[2][0])
        remove_accuracy.append(data[2][1])

    color_list = ["green", "yellow", "red"]
    color_map = LinearSegmentedColormap.from_list(
        name="green_to_red", colors=color_list
    )

    fig = plt.figure()

    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    main_graph = fig.add_axes([left, bottom, width, height])
    main_graph.set_xlabel("A$_s$")
    main_graph.set_ylabel("A$_r$")

    main_graph.scatter(
        separate_accuracy,
        remove_accuracy,
        c=range(len(remove_accuracy)),
        cmap=color_map,
        s=0.5,
    )

    position = {"MNIST": (1.25, -1, -6, -3), "EMNIST": (2.3, 0, -10, -3)}
    # best
    best_annotation = {
        "MNIST": "best\nA$_s$:99.26\nA$_r$:77.32",
        "EMNIST": "best\nA$_s$:92.55\nA$_r$:76.31",
    }
    main_graph.scatter(
        separate_accuracy[0],
        remove_accuracy[0],
        color="green",
        s=10,
    )
    main_graph.annotate(
        text=best_annotation[dataset],
        xy=(separate_accuracy[0], remove_accuracy[0]),
        xytext=(
            separate_accuracy[0] + position[dataset][0],
            remove_accuracy[0] + position[dataset][1],
        ),
        arrowprops={"arrowstyle": "->"},
    )

    # worst
    worst_annotation = {
        "MNIST": "worst\nA$_s$:92.12\nA$_r$:93.44",
        "EMNIST": "worst\nA$_s$:68.92\nA$_r$:66.88",
    }
    main_graph.scatter(
        separate_accuracy[-1],
        remove_accuracy[-1],
        color="red",
        s=10,
    )
    main_graph.annotate(
        text=worst_annotation[dataset],
        xy=(separate_accuracy[-1], remove_accuracy[-1]),
        xytext=(
            separate_accuracy[-1] + position[dataset][2],
            remove_accuracy[-1] + position[dataset][3],
        ),
        arrowprops={"arrowstyle": "->"},
    )

    main_graph.set_title(dataset)

    # show colormap
    left, bottom, width, height = 0.125, 0.825, 0.2, 0.05
    color_map_graph = fig.add_axes([left, bottom, width, height])

    norm = mpl.colors.Normalize(vmin=1, vmax=len(data_source))
    color_list = ["green", "yellow", "red"]
    color_map = LinearSegmentedColormap.from_list(
        name="green_to_red", colors=color_list
    )

    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=color_map),
        cax=color_map_graph,
        orientation="horizontal",
        label="Ranking",
        ticks=[1, len(data_source)],
    )

    plt.savefig(f"{dataset}_distribution.png")
    plt.close()


def draw_accuracy_distribution_graph(MNIST_filtered_data, EMNIST_filtered_data):
    draw_accuracy_distribution("MNIST", MNIST_filtered_data)
    draw_accuracy_distribution("EMNIST", EMNIST_filtered_data)


# calculate_ranking
data_path = {
    "MNIST": "../full_combinations/MNIST.npy",
    "EMNIST": "../full_combinations/EMNIST.npy",
}
MNIST_ranking = calculate_selection_ranking("MNIST", data_path["MNIST"])
EMNIST_ranking = calculate_selection_ranking("EMNIST", data_path["EMNIST"])

# extract data
data_categoty = {"MNIST": (2, 6), "EMNIST": (2, 4)}
MNIST_filtered_data = extract_data(MNIST_ranking, data_categoty["MNIST"])
EMNIST_filtered_data = extract_data(EMNIST_ranking, data_categoty["EMNIST"])

# draw the line
draw_ranking_accuracy_graph(MNIST_filtered_data, EMNIST_filtered_data)
draw_accuracy_distribution_graph(MNIST_filtered_data, EMNIST_filtered_data)

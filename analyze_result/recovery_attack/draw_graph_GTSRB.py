import matplotlib.pyplot as plt
import numpy as np


def draw_recovery_attack_graph(
    percents,
    separate_accuracies,
    remove_accuracies,
    # recover_accuracies_freeze,
    recover_accuracies_unfreeze,
    running_time,
):
    figure = plt.figure(figsize=(7, 4))

    figure1 = figure.add_subplot(111)
    figure1.set_xlabel("Selected Neurons (%)")
    figure1.set_ylabel("Accuracy (%)")

    figure1.set_xticks(np.linspace(50, 100, 11))  # [50, 105, 5]
    figure1.set_yticks(np.linspace(10, 100, 10))  # [10, 110, 10]

    separate_line = figure1.plot(
        percents,
        separate_accuracies,
        linestyle="solid",
        color="green",
        label="$A_s$",
    )

    remove_line = figure1.plot(
        percents,
        remove_accuracies,
        linestyle="solid",
        color="orange",
        label="$A_r$",
    )

    recover_line = figure1.plot(
        percents,
        recover_accuracies_unfreeze,
        linestyle="solid",
        color="purple",
        label="$A_{rec}$",
    )

    # train from scratch
    train_line = figure1.axhline(
        91.45, linestyle="solid", color="blue", label="$A_t$", alpha=0.3
    )
    figure1.scatter(80, 91.45, color="blue", zorder=2, s=15)
    figure1.annotate(
        f"{91.45:.2f}",
        xy=(80, 91.45),
        xytext=(80 + 1, 91.45 - 4.5),
        color="blue",
    )

    figure2 = figure1.twinx()
    plt.ylim(0, 300)
    figure2.set_ylabel("T (1000 seconds)")
    figure2.set_yticks(ticks=(0, 100, 200, 300))
    time_line = figure2.plot(
        percents,
        running_time,
        linestyle="solid",
        color="salmon",
        label="$T$",
    )

    figure1.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.015))
    figure2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.015))

    # separate accuracy at 50%, 75% and 100%
    figure1.scatter(percents[0], separate_accuracies[0], color="green", zorder=1, s=15)
    figure1.annotate(
        f"{separate_accuracies[0]:.2f}",
        xy=(percents[0], separate_accuracies[0]),
        xytext=(percents[0] - 2, separate_accuracies[0] + 2),
        color="green",
    )
    figure1.scatter(percents[5], separate_accuracies[5], color="green", zorder=1, s=15)
    figure1.annotate(
        f"{separate_accuracies[5]:.2f}",
        xy=(percents[5], separate_accuracies[5]),
        xytext=(percents[5] - 5, separate_accuracies[5] - 9),
        color="green",
    )
    figure1.scatter(
        percents[10], separate_accuracies[10], color="green", zorder=1, s=15
    )
    figure1.annotate(
        f"{separate_accuracies[10]:.2f}",
        xy=(percents[10], separate_accuracies[10]),
        xytext=(percents[10] - 2, separate_accuracies[10] + 2),
        color="green",
    )

    # remove accuracy at 50%, 75% and 100%
    figure1.scatter(percents[0], remove_accuracies[0], color="orange", zorder=1, s=15)
    figure1.annotate(
        f"{remove_accuracies[0]:.2f}",
        xy=(percents[0], remove_accuracies[0]),
        xytext=(percents[0] - 2, remove_accuracies[0] + 2),
        color="orange",
    )
    figure1.scatter(percents[5], remove_accuracies[5], color="orange", zorder=1, s=15)
    figure1.annotate(
        f"{remove_accuracies[5]:.2f}",
        xy=(percents[5], remove_accuracies[5]),
        xytext=(percents[5] - 3, remove_accuracies[5] - 6),
        color="orange",
    )
    figure1.scatter(percents[10], remove_accuracies[10], color="orange", zorder=1, s=15)
    figure1.annotate(
        f"{remove_accuracies[10]:.2f}",
        xy=(percents[10], remove_accuracies[10]),
        xytext=(percents[10] - 3.5, remove_accuracies[10] - 4),
        color="orange",
    )

    # recover accuracy at 50%, 75% and 100%
    figure1.scatter(
        percents[0], recover_accuracies_unfreeze[0], color="purple", zorder=1, s=15
    )
    figure1.annotate(
        f"{recover_accuracies_unfreeze[0]:.2f}",
        xy=(percents[0], recover_accuracies_unfreeze[0]),
        xytext=(percents[0] - 2, recover_accuracies_unfreeze[0] - 5),
        color="purple",
    )
    figure1.scatter(
        percents[5], recover_accuracies_unfreeze[5], color="purple", zorder=1, s=15
    )
    figure1.annotate(
        f"{recover_accuracies_unfreeze[5]:.2f}",
        xy=(percents[5], recover_accuracies_unfreeze[5]),
        xytext=(percents[5], recover_accuracies_unfreeze[5] - 7.5),
        color="purple",
    )
    figure1.scatter(
        percents[10], recover_accuracies_unfreeze[10], color="purple", zorder=1, s=15
    )
    figure1.annotate(
        f"{recover_accuracies_unfreeze[10]:.2f}",
        xy=(percents[10], recover_accuracies_unfreeze[10]),
        xytext=(percents[10] - 4, recover_accuracies_unfreeze[10]),
        color="purple",
    )

    # running time at 50%, 75% and 100%
    figure2.scatter(percents[0], running_time[0], color="salmon", zorder=1, s=15)
    figure2.annotate(
        f"{running_time[0]:.2f}",
        xy=(percents[0], running_time[0]),
        xytext=(percents[0] - 2, running_time[0] + 6),
        color="salmon",
    )
    figure2.scatter(percents[5], running_time[5], color="salmon", zorder=1, s=15)
    figure2.annotate(
        f"{running_time[5]:.2f}",
        xy=(percents[5], running_time[5]),
        xytext=(percents[5] - 2, running_time[5] - 15),
        color="salmon",
    )
    figure2.scatter(percents[10], running_time[10], color="salmon", zorder=1, s=15)
    figure2.annotate(
        f"{running_time[10]:.2f}",
        xy=(percents[10], running_time[10]),
        xytext=(percents[10] - 2.5, running_time[10] + 7),
        color="salmon",
    )

    for i in range(10, 110, 10):
        figure1.axhline(i, linestyle="--", color="gray", alpha=0.3)

    plt.savefig(f"recovery_attack_GTSRB.png", dpi=1024)
    plt.close()


if __name__ == "__main__":
    percents = [*range(50, 105, 5)]
    separate_accuracies = [
        94.27,  # 50
        94.25,  # 55
        94.19,  # 60
        94.19,  # 65
        94.27,  # 70
        94.25,  # 75
        94.24,  # 80
        94.35,  # 85
        94.16,  # 90
        94.08,  # 95
        93.51,  # 100
    ]
    remove_accuracies = [
        68.95,  # 50
        62.44,  # 55
        55.72,  # 60
        50.66,  # 65
        44.59,  # 70
        37.96,  # 75
        29.74,  # 80
        22.95,  # 85
        11.40,  # 90
        3.78,  # 95
        1.01,  # 100
    ]
    recover_accuracies_freeze = [
        66.41,  # 50
        61.54,  # 55
        53.63,  # 60
        48.57,  # 65
        44.09,  # 70
        34.87,  # 75
        27.70,  # 80
        21.25,  # 85
        11.83,  # 90
        5.99,  # 95
        1.66,  # 100
    ]
    recover_accuracies_unfreeze = [
        92.70,  # 50
        92.59,  # 55
        91.89,  # 60
        91.40,  # 65
        90.63,  # 70
        90.40,  # 75
        86.32,  # 80
        85.68,  # 85
        80.35,  # 90
        74.60,  # 95
        5.42,  # 100
    ]
    running_time = [
        54185,  # 50
        58768,  # 55
        63378,  # 60
        68500,  # 65
        73599,  # 70
        79268,  # 75
        85399,  # 80
        90701,  # 85
        95889,  # 90
        101028,  # 95
        108762,  # 100
    ]

    draw_recovery_attack_graph(
        percents,
        separate_accuracies,
        remove_accuracies,
        # recover_accuracies_freeze,
        recover_accuracies_unfreeze,
        [i / 1000 for i in running_time],
    )

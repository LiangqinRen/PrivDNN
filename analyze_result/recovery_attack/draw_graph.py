import sys
import matplotlib.pyplot as plt
import numpy as np


def draw_recovery_attack_graph(
    percents,
    separate_accuracies,
    remove_accuracies,
    recover_accuracies,
    running_time,
):
    figure = plt.figure(figsize=(7, 4))

    figure1 = figure.add_subplot(111)
    figure1.set_xlabel("Selected Neurons (%)")
    figure1.set_ylabel("%")

    figure1.set_xticks(np.linspace(50, 100, 11))  # [50, 105, 5]
    figure1.set_yticks(np.linspace(10, 100, 10))  # [10, 110, 10]

    figure1.plot(
        percents,
        separate_accuracies,
        linestyle="solid",
        color="green",
        label="$A_s$",
    )

    figure1.plot(
        percents,
        remove_accuracies,
        linestyle="solid",
        color="orange",
        label="$A_r$",
    )

    figure1.plot(
        percents,
        recover_accuracies,
        linestyle="solid",
        color="brown",
        label="$A_{rec}$",
    )
    figure1.legend(loc="center left")

    figure2 = figure1.twinx()
    figure2.set_ylim(ymin=50, ymax=150)
    figure2.set_yticks(np.linspace(50, 150, 11))
    figure2.plot(
        percents,
        running_time,
        linestyle="solid",
        color="purple",
        label="$T$",
    )
    figure2.set_ylabel("1000 seconds")
    figure2.legend(loc="center right")

    figure1.scatter(percents[4], separate_accuracies[4], color="red", zorder=2)
    figure1.annotate(
        f"{separate_accuracies[4]:.2f}",
        xy=(percents[4], separate_accuracies[5]),
        xytext=(percents[4] + 1, separate_accuracies[5] + 1),
        color="red",
    )
    figure1.scatter(percents[4], remove_accuracies[4], color="red", zorder=2)
    figure1.annotate(
        f"{remove_accuracies[4]:.2f}",
        xy=(percents[4], remove_accuracies[4]),
        xytext=(percents[4] - 1, remove_accuracies[4] - 7),
        color="red",
    )
    figure1.scatter(percents[4], recover_accuracies[4], color="red", zorder=2)
    figure1.annotate(
        f"{recover_accuracies[4]:.2f}",
        xy=(percents[4], recover_accuracies[4]),
        xytext=(percents[4] + 1, recover_accuracies[4] + 1),
        color="red",
    )
    figure2.scatter(percents[4], running_time[4], color="red", zorder=2)
    figure2.annotate(
        f"{running_time[4]:.2f}",
        xy=(percents[4], running_time[4]),
        xytext=(percents[4] + 2, running_time[4] - 1),
        color="red",
    )

    for i in range(10, 110, 10):
        figure1.axhline(i, linestyle="--", color="gray", alpha=0.3)

    plt.savefig(f"recovery_attack.png", dpi=1024)
    plt.close()


if __name__ == "__main__":
    percents = [*range(50, 105, 5)]
    separate_accuracies = [
        90.48,
        90.40,
        90.50,
        90.54,
        90.56,
        90.46,
        90.60,
        90.44,
        90.46,
        90.30,
        90.32,
    ]
    remove_accuracies = [
        20.16,
        17.50,
        15.08,
        16.36,
        16.44,
        14.48,
        11.34,
        9.90,
        9.88,
        9.78,
        9.78,
    ]
    recover_accuracies = [
        83.36,
        79.92,
        79.12,
        65.32,
        80.12,
        68.20,
        43.68,
        25.24,
        24.32,
        21.64,
        22.60,
    ]
    running_time = [31097, 33723, 36316, 39310, 0, 0, 0, 0, 0, 0, 0]

    draw_recovery_attack_graph(
        percents,
        separate_accuracies,
        remove_accuracies,
        recover_accuracies,
        [i / 1000 for i in running_time],
    )

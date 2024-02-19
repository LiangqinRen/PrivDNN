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
        10.14, linestyle="solid", color="blue", label="$A_t$", alpha=0.3
    )
    figure1.scatter(80, 10.14, color="blue", zorder=2, s=15)
    figure1.annotate(
        f"{10.14:.2f}",
        xy=(80, 10.14),
        xytext=(80, 10.14 + 2),
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
        xytext=(percents[5] - 2, separate_accuracies[5] - 5),
        color="green",
    )
    figure1.scatter(
        percents[10], separate_accuracies[10], color="green", zorder=1, s=15
    )
    figure1.annotate(
        f"{separate_accuracies[10]:.2f}",
        xy=(percents[10], separate_accuracies[10]),
        xytext=(percents[10] - 2, separate_accuracies[10] - 5),
        color="green",
    )

    # remove accuracy at 50%, 75% and 100%
    figure1.scatter(percents[0], remove_accuracies[0], color="orange", zorder=1, s=15)
    figure1.annotate(
        f"{remove_accuracies[0]:.2f}",
        xy=(percents[0], remove_accuracies[0]),
        xytext=(percents[0] - 2, remove_accuracies[0] - 6),
        color="orange",
    )
    figure1.scatter(percents[5], remove_accuracies[5], color="orange", zorder=1, s=15)
    figure1.annotate(
        f"{remove_accuracies[5]:.2f}",
        xy=(percents[5], remove_accuracies[5]),
        xytext=(percents[5] - 2, remove_accuracies[5] - 7),
        color="orange",
    )
    figure1.scatter(percents[10], remove_accuracies[10], color="orange", zorder=1, s=15)
    figure1.annotate(
        f"{remove_accuracies[10]:.2f}",
        xy=(percents[10], remove_accuracies[10]),
        xytext=(percents[10] - 4, remove_accuracies[10] - 4),
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
        xytext=(percents[5] - 2, recover_accuracies_unfreeze[5] - 5),
        color="purple",
    )
    figure1.scatter(
        percents[10], recover_accuracies_unfreeze[10], color="purple", zorder=1, s=15
    )
    figure1.annotate(
        f"{recover_accuracies_unfreeze[10]:.2f}",
        xy=(percents[10], recover_accuracies_unfreeze[10]),
        xytext=(percents[10] - 0.7, recover_accuracies_unfreeze[10] + 1.5),
        color="purple",
    )

    # running time at 50%, 75% and 100%
    figure2.scatter(percents[0], running_time[0], color="salmon", zorder=1, s=15)
    figure2.annotate(
        f"{running_time[0]:.2f}",
        xy=(percents[0], running_time[0]),
        xytext=(percents[0] - 2, running_time[0] + 8),
        color="salmon",
    )
    figure2.scatter(percents[5], running_time[5], color="salmon", zorder=1, s=15)
    figure2.annotate(
        f"{running_time[5]:.2f}",
        xy=(percents[5], running_time[5]),
        xytext=(percents[5] - 3, running_time[5] + 8),
        color="salmon",
    )
    figure2.scatter(percents[10], running_time[10], color="salmon", zorder=1, s=15)
    figure2.annotate(
        f"{running_time[10]:.2f}",
        xy=(percents[10], running_time[10]),
        xytext=(percents[10] - 3, running_time[10] - 25),
        color="salmon",
    )

    for i in range(10, 110, 10):
        figure1.axhline(i, linestyle="--", color="gray", alpha=0.3)

    plt.savefig(f"recovery_attack_TinyImageNet.png", dpi=1024)
    plt.close()


if __name__ == "__main__":
    percents = [*range(50, 105, 5)]
    separate_accuracies = [
        72.02,  # 50
        72.14,  # 55
        72.06,  # 60
        72.22,  # 65
        71.92,  # 70
        72.04,  # 75
        72.02,  # 80
        72.06,  # 85
        72.14,  # 90
        72.00,  # 95
        72.00,  # 100
    ]
    remove_accuracies = [
        45.66,  # 50
        41.04,  # 55
        38.78,  # 60
        32.22,  # 65
        32.78,  # 70
        26.76,  # 75
        19.16,  # 80
        17.36,  # 85
        10.00,  # 90
        4.94,  # 95
        2.50,  # 100
    ]
    recover_accuracies_freeze = [
        0,  # 50
        0,  # 55
        0,  # 60
        0,  # 65
        0,  # 70
        0,  # 75
        0,  # 80
        0,  # 85
        0,  # 90
        0,  # 95
        0,  # 100
    ]
    recover_accuracies_unfreeze = [
        66.16,  # 50
        66.04,  # 55
        65.66,  # 60
        64.76,  # 65
        63.58,  # 70
        61.88,  # 75
        59.10,  # 80
        56.86,  # 85
        45.14,  # 90
        15.44,  # 95
        4.20,  # 100
    ]
    running_time = [
        143300,  # 50
        155800,  # 55
        168506,  # 60
        184363,  # 65
        198387,  # 70
        213730,  # 75
        231486,  # 80
        247753,  # 85
        263162,  # 90
        278112,  # 95
        297387,  # 100
    ]

    draw_recovery_attack_graph(
        percents,
        separate_accuracies,
        remove_accuracies,
        # recover_accuracies_freeze,
        recover_accuracies_unfreeze,
        [i / 1000 for i in running_time],
    )

import matplotlib.pyplot as plt
import numpy as np


def draw_recovery_attack_graph(
    percents,
    separate_accuracies,
    remove_accuracies,
    recover_accuracies_freeze,
    recover_accuracies_unfreeze,
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

    """figure1.plot(
        percents,
        recover_accuracies_freeze,
        linestyle="solid",
        color="red",
        label="$A_{rec}$ ✓",
    )
    figure1.legend(loc="center left")"""

    figure1.plot(
        percents,
        recover_accuracies_unfreeze,
        linestyle="solid",
        color="purple",
        label="$A_{rec}$",
    )

    # train from scratch
    figure1.axhline(49.94, linestyle="solid", color="blue", label="$A_o$", alpha=0.3)
    figure1.scatter(90, 49.94, color="red", zorder=2)
    figure1.annotate(
        f"{49.94:.2f}",
        xy=(90, 49.94),
        xytext=(90, 49.94 + 2),
        color="red",
    )

    figure1.legend(loc="center left")

    figure2 = figure1.twinx()
    figure2.set_ylabel("✖️ 1000 seconds")
    figure2.set_yticks(np.linspace(10, 100, 10))
    figure2.plot(
        percents,
        running_time,
        linestyle="solid",
        color="coral",
        label="$T$",
    )

    figure2.legend(loc="center right")

    figure1.scatter(percents[6], separate_accuracies[6], color="red", zorder=2)
    figure1.annotate(
        f"{separate_accuracies[6]:.2f}",
        xy=(percents[6], separate_accuracies[6]),
        xytext=(percents[6] + 1, separate_accuracies[6] + 1),
        color="red",
    )
    figure1.scatter(percents[6], remove_accuracies[6], color="red", zorder=2)
    figure1.annotate(
        f"{remove_accuracies[6]:.2f}",
        xy=(percents[6], remove_accuracies[6]),
        xytext=(percents[6] - 3, remove_accuracies[6] - 6),
        color="red",
    )
    """figure1.scatter(percents[5], recover_accuracies_freeze[5], color="red", zorder=2)
    figure1.annotate(
        f"{recover_accuracies_freeze[5]:.2f}",
        xy=(percents[5], recover_accuracies_freeze[5]),
        xytext=(percents[5] - 5, recover_accuracies_freeze[5]),
        color="red",
    )"""
    figure1.scatter(percents[6], recover_accuracies_unfreeze[6], color="red", zorder=2)
    figure1.annotate(
        f"{recover_accuracies_unfreeze[6]:.2f}",
        xy=(percents[6], recover_accuracies_unfreeze[6]),
        xytext=(percents[6] + 1, recover_accuracies_unfreeze[6] + 1),
        color="red",
    )

    figure2.scatter(percents[6], running_time[6], color="red", zorder=2)
    figure2.annotate(
        f"{running_time[6]:.2f}",
        xy=(percents[6], running_time[6]),
        xytext=(percents[6] - 3, running_time[6] + 1),
        color="red",
    )

    for i in range(10, 110, 10):
        figure1.axhline(i, linestyle="--", color="gray", alpha=0.3)

    plt.savefig(f"recovery_attack.png", dpi=1024)
    plt.close()


if __name__ == "__main__":
    percents = [*range(50, 105, 5)]
    separate_accuracies = [
        90.78,  # 50
        90.76,  # 55
        90.64,  # 60
        90.54,  # 65
        90.68,  # 70
        90.52,  # 75
        90.56,  # 80
        90.46,  # 85
        90.26,  # 90
        90.48,  # 95
        90.22,  # 100
    ]
    remove_accuracies = [
        55.90,  # 50
        52.50,  # 55
        43.68,  # 60
        44.06,  # 65
        43.74,  # 70
        24.90,  # 75
        18.48,  # 80
        13.38,  # 85
        9.82,  # 90
        9.78,  # 95
        9.78,  # 100
    ]
    recover_accuracies_freeze = [
        82.74,  # 50
        82.34,  # 55
        79.90,  # 60
        78.92,  # 65
        77.70,  # 70
        21.88,  # 75
        24.62,  # 80
        24.90,  # 85
        23.30,  # 90
        24.16,  # 95
        20.82,  # 100
    ]
    recover_accuracies_unfreeze = [
        83.60,  # 50
        83.24,  # 55
        80.90,  # 60
        79.66,  # 65
        78.66,  # 70
        60.26,  # 75
        43.14,  # 80
        39.42,  # 85
        34.12,  # 90
        35.02,  # 95
        30.18,  # 100
    ]
    # running_time = [31097, 33723, 36316, 39310, 42210, 45378, 48940, 0, 0, 0, 0]
    running_time = [
        31097,  # 50
        33723,  # 55
        36316,  # 60
        39310,  # 65
        42210,  # 70
        45378,  # 75
        48940,  # 80
        51940,  # 85
        54940,  # 90
        57940,  # 95
        60940,  # 100
    ]

    draw_recovery_attack_graph(
        percents,
        separate_accuracies,
        remove_accuracies,
        recover_accuracies_freeze,
        recover_accuracies_unfreeze,
        [i / 1000 for i in running_time],
    )

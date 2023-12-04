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

    figure1.plot(
        percents,
        recover_accuracies_freeze,
        linestyle="solid",
        color="red",
        label="$A_{rec}$ ✓",
    )
    figure1.legend(loc="center left")

    figure1.plot(
        percents,
        recover_accuracies_unfreeze,
        linestyle="solid",
        color="brown",
        label="$A_{rec}$ ✕",
    )

    # train from scratch
    figure1.axhline(40.20, linestyle="solid", color="blue", label="$A_o$", alpha=0.3)
    figure1.scatter(80, 40.20, color="red", zorder=2)
    figure1.annotate(
        f"{40.20:.2f}",
        xy=(80, 40.20),
        xytext=(80, 40.20 + 2),
        color="red",
    )

    figure1.legend(loc="center left")

    figure2 = figure1.twinx()
    figure2.set_ylabel("✖️ 1000 seconds")
    # figure2.set_ylim(ymin=0, ymax=100)
    figure2.set_yticks(np.linspace(10, 100, 10))
    figure2.plot(
        percents,
        running_time,
        linestyle="solid",
        color="purple",
        label="$T$",
    )

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
        xytext=(percents[4] + 2, remove_accuracies[4] + 2),
        color="red",
    )
    figure1.scatter(percents[3], recover_accuracies_freeze[3], color="red", zorder=2)
    figure1.annotate(
        f"{recover_accuracies_freeze[3]:.2f}",
        xy=(percents[3], recover_accuracies_freeze[3]),
        xytext=(percents[3] + 1, recover_accuracies_freeze[3] + 1),
        color="red",
    )
    figure1.scatter(percents[4], recover_accuracies_unfreeze[4], color="red", zorder=2)
    figure1.annotate(
        f"{recover_accuracies_unfreeze[4]:.2f}",
        xy=(percents[4], recover_accuracies_unfreeze[4]),
        xytext=(percents[4] + 1, recover_accuracies_unfreeze[4] + 2),
        color="red",
    )

    figure2.scatter(percents[4], running_time[4], color="red", zorder=2)
    figure2.annotate(
        f"{running_time[4]:.2f}",
        xy=(percents[4], running_time[4]),
        xytext=(percents[4] - 1, running_time[4] + 2),
        color="red",
    )

    for i in range(10, 110, 10):
        figure1.axhline(i, linestyle="--", color="gray", alpha=0.3)

    plt.savefig(f"recovery_attack.png", dpi=1024)
    plt.close()


if __name__ == "__main__":
    percents = [*range(50, 105, 5)]
    separate_accuracies = [
        89.90,  # 50
        89.90,  # 55
        89.62,  # 60
        89.66,  # 65
        89.66,  # 70
        89.72,  # 75
        89.50,  # 80
        89.44,  # 85
        89.44,  # 90
        89.38,  # 95
        89.56,  # 100
    ]
    remove_accuracies = [
        43.78,  # 50
        28.88,  # 55
        21.80,  # 60
        17.02,  # 65
        11.00,  # 70
        10.80,  # 75
        10.12,  # 80
        10.00,  # 85
        10.12,  # 90
        10.02,  # 95
        10.24,  # 100
    ]
    recover_accuracies_freeze = [
        84.44,  # 50
        82.14,  # 55
        69.82,  # 60
        19.48,  # 65
        18.26,  # 70
        19.46,  # 75
        19.20,  # 80
        20.92,  # 85
        19.42,  # 90
        20.14,  # 95
        16.02,  # 100
    ]
    recover_accuracies_unfreeze = [
        84.66,  # 50
        83.04,  # 55
        77.94,  # 60
        79.32,  # 65
        26.44,  # 70
        27.02,  # 75
        26.42,  # 80
        25.82,  # 85
        33.96,  # 90
        29.76,  # 95
        25.92,  # 100
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

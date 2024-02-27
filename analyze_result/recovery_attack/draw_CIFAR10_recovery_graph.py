import matplotlib.pyplot as plt
import numpy as np


def draw_recovery_attack_graph(
    percents,
    separate_accuracies,
    remove_accuracies,
    # recover_accuracies_freeze,
    recover_accuracies_unfreeze,
):
    figure = plt.figure(figsize=(6.5, 5))

    figure1 = figure.add_subplot(111)
    figure1.set_xlabel("Selected Neurons (%)", fontsize=15)
    figure1.set_ylabel("Accuracy (%)", fontsize=15)

    figure1.set_xticks(np.linspace(50, 100, 11))  # [50, 105, 5]
    figure1.set_yticks(np.linspace(10, 100, 10))  # [10, 110, 10]
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

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
        recover_accuracies_unfreeze,
        linestyle="solid",
        color="purple",
        label="$A_{rec}$",
    )

    # train from scratch
    figure1.axhline(49.94, linestyle="solid", color="blue", label="$A_t$", alpha=0.3)
    figure1.scatter(80, 49.94, color="blue", zorder=2, s=20)
    figure1.annotate(
        f"{49.94:.2f}",
        xy=(80, 49.94),
        xytext=(80, 49.94 + 2),
        color="blue",
        fontsize=15,
    )

    figure1.legend(
        loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.015), fontsize="15"
    )

    # separate accuracy at 50%, 75% and 100%
    figure1.scatter(percents[0], separate_accuracies[0], color="green", zorder=1, s=20)
    figure1.annotate(
        f"{separate_accuracies[0]:.2f}",
        xy=(percents[0], separate_accuracies[0]),
        xytext=(percents[0] - 2, separate_accuracies[0] - 5),
        color="green",
        fontsize=15,
    )
    figure1.scatter(percents[5], separate_accuracies[5], color="green", zorder=1, s=20)
    figure1.annotate(
        f"{separate_accuracies[5]:.2f}",
        xy=(percents[5], separate_accuracies[5]),
        xytext=(percents[5] - 3, separate_accuracies[5] - 5),
        color="green",
        fontsize=15,
    )
    figure1.scatter(
        percents[10], separate_accuracies[10], color="green", zorder=1, s=20
    )
    figure1.annotate(
        f"{separate_accuracies[10]:.2f}",
        xy=(percents[10], separate_accuracies[10]),
        xytext=(percents[10] - 5, separate_accuracies[10] - 5),
        color="green",
        fontsize=15,
    )

    # remove accuracy at 50%, 75% and 100%
    figure1.scatter(percents[0], remove_accuracies[0], color="orange", zorder=1, s=20)
    figure1.annotate(
        f"{remove_accuracies[0]:.2f}",
        xy=(percents[0], remove_accuracies[0]),
        xytext=(percents[0] - 2, remove_accuracies[0] + 2),
        color="orange",
        fontsize=15,
    )
    figure1.scatter(percents[5], remove_accuracies[5], color="orange", zorder=1, s=20)
    figure1.annotate(
        f"{remove_accuracies[5]:.2f}",
        xy=(percents[5], remove_accuracies[5]),
        xytext=(percents[5] + 0.5, remove_accuracies[5] + 1),
        color="orange",
        fontsize=15,
    )
    figure1.scatter(percents[10], remove_accuracies[10], color="orange", zorder=1, s=20)
    figure1.annotate(
        f"{remove_accuracies[10]:.2f}",
        xy=(percents[10], remove_accuracies[10]),
        xytext=(percents[10] - 3, remove_accuracies[10] + 2),
        color="orange",
        fontsize=15,
    )

    # recover accuracy at 50%, 75% and 100%
    figure1.scatter(
        percents[0], recover_accuracies_unfreeze[0], color="purple", zorder=1, s=20
    )
    figure1.annotate(
        f"{recover_accuracies_unfreeze[0]:.2f}",
        xy=(percents[0], recover_accuracies_unfreeze[0]),
        xytext=(percents[0] - 2, recover_accuracies_unfreeze[0] - 5),
        color="purple",
        fontsize=15,
    )
    figure1.scatter(
        percents[5], recover_accuracies_unfreeze[5], color="purple", zorder=1, s=20
    )
    figure1.annotate(
        f"{recover_accuracies_unfreeze[5]:.2f}",
        xy=(percents[5], recover_accuracies_unfreeze[5]),
        xytext=(percents[5] - 7, recover_accuracies_unfreeze[5] - 1.5),
        color="purple",
        fontsize=15,
    )
    figure1.scatter(
        percents[10], recover_accuracies_unfreeze[10], color="purple", zorder=1, s=20
    )
    figure1.annotate(
        f"{recover_accuracies_unfreeze[10]:.2f}",
        xy=(percents[10], recover_accuracies_unfreeze[10]),
        xytext=(percents[10] - 5, recover_accuracies_unfreeze[10] - 5),
        color="purple",
        fontsize=15,
    )

    for i in range(10, 110, 10):
        figure1.axhline(i, linestyle="--", color="gray", alpha=0.3)

    plt.savefig(f"recovery_attack_CIFAR10.png", dpi=1024)
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
        58.20,  # 75
        43.14,  # 80
        39.42,  # 85
        34.12,  # 90
        35.02,  # 95
        30.18,  # 100
    ]

    draw_recovery_attack_graph(
        percents,
        separate_accuracies,
        remove_accuracies,
        # recover_accuracies_freeze,
        recover_accuracies_unfreeze,
    )

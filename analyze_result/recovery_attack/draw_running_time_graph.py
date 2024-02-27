import matplotlib.pyplot as plt
import numpy as np


def draw_running_time_graph(
    percents,
    cifar10_running_time,
    gtsrb_running_time,
    tinyimagenet_running_time,
):
    figure = plt.figure(figsize=(6.5, 5))

    figure1 = figure.add_subplot(111)
    figure1.set_xlabel("Selected Neurons (%)", fontsize=15)
    figure1.set_ylabel("Model Evaluation Time (1,000 seconds)", fontsize=15)

    figure1.set_xticks(np.linspace(50, 100, 11))  # [50, 105, 5]
    figure1.set_yticks(ticks=(0, 100, 200, 300))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    figure1.plot(
        percents,
        cifar10_running_time,
        linestyle="solid",
        color="royalblue",
        label="CIFAR10",
    )

    figure1.plot(
        percents,
        gtsrb_running_time,
        linestyle="solid",
        color="teal",
        label="GTSRB",
    )

    figure1.plot(
        percents,
        tinyimagenet_running_time,
        linestyle="solid",
        color="firebrick",
        label="Tiny-ImageNet",
    )

    figure1.legend(loc="center right", fontsize="15")

    # CIFAR10 running time at 50%, 75% and 100%
    figure1.scatter(
        percents[0], cifar10_running_time[0], color="royalblue", zorder=1, s=15
    )
    figure1.annotate(
        f"{cifar10_running_time[0]:.2f}",
        xy=(percents[0], cifar10_running_time[0]),
        xytext=(percents[0] - 2, cifar10_running_time[0] - 17.5),
        color="royalblue",
        fontsize=15,
    )
    figure1.scatter(
        percents[5], cifar10_running_time[5], color="royalblue", zorder=1, s=15
    )
    figure1.annotate(
        f"{cifar10_running_time[5]:.2f}",
        xy=(percents[5], cifar10_running_time[5]),
        xytext=(percents[5] - 2, cifar10_running_time[5] - 17.5),
        color="royalblue",
        fontsize=15,
    )
    figure1.scatter(
        percents[10], cifar10_running_time[10], color="royalblue", zorder=1, s=15
    )
    figure1.annotate(
        f"{cifar10_running_time[10]:.2f}",
        xy=(percents[10], cifar10_running_time[10]),
        xytext=(percents[10] - 5, cifar10_running_time[10] - 17.5),
        color="royalblue",
        fontsize=15,
    )

    # GTSRB running time at 50%, 75% and 100%
    figure1.scatter(percents[0], gtsrb_running_time[0], color="teal", zorder=1, s=15)
    figure1.annotate(
        f"{gtsrb_running_time[0]:.2f}",
        xy=(percents[0], gtsrb_running_time[0]),
        xytext=(percents[0] - 2, gtsrb_running_time[0] + 8),
        color="teal",
        fontsize=15,
    )
    figure1.scatter(percents[5], gtsrb_running_time[5], color="teal", zorder=1, s=15)
    figure1.annotate(
        f"{gtsrb_running_time[5]:.2f}",
        xy=(percents[5], gtsrb_running_time[5]),
        xytext=(percents[5] - 2, gtsrb_running_time[5] + 7),
        color="teal",
        fontsize=15,
    )
    figure1.scatter(percents[10], gtsrb_running_time[10], color="teal", zorder=1, s=15)
    figure1.annotate(
        f"{gtsrb_running_time[10]:.2f}",
        xy=(percents[10], gtsrb_running_time[10]),
        xytext=(percents[10] - 6, gtsrb_running_time[10] - 22),
        color="teal",
        fontsize=15,
    )

    # TinyImageNet running time at 50%, 75% and 100%
    figure1.scatter(
        percents[0], tinyimagenet_running_time[0], color="firebrick", zorder=1, s=15
    )
    figure1.annotate(
        f"{tinyimagenet_running_time[0]:.2f}",
        xy=(percents[0], tinyimagenet_running_time[0]),
        xytext=(percents[0] - 2, tinyimagenet_running_time[0] - 17.5),
        color="firebrick",
        fontsize=15,
    )
    figure1.scatter(
        percents[5], tinyimagenet_running_time[5], color="firebrick", zorder=1, s=15
    )
    figure1.annotate(
        f"{tinyimagenet_running_time[5]:.2f}",
        xy=(percents[5], tinyimagenet_running_time[5]),
        xytext=(percents[5] + 1, tinyimagenet_running_time[5] - 9),
        color="firebrick",
        fontsize=15,
    )
    figure1.scatter(
        percents[10], tinyimagenet_running_time[10], color="firebrick", zorder=1, s=15
    )
    figure1.annotate(
        f"{tinyimagenet_running_time[10]:.2f}",
        xy=(percents[10], tinyimagenet_running_time[10]),
        xytext=(percents[10] - 9, tinyimagenet_running_time[10] - 5),
        color="firebrick",
        fontsize=15,
    )

    for i in range(0, 300, 30):
        figure1.axhline(i, linestyle="--", color="gray", alpha=0.3)

    plt.savefig(f"running_time.png", dpi=1024)
    plt.close()


if __name__ == "__main__":
    percents = [*range(50, 105, 5)]
    cifar10_running_time = [
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

    gtsrb_running_time = [
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

    tinyimagenet_running_time = [
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

    draw_running_time_graph(
        percents,
        [i / 1000 for i in cifar10_running_time],
        [i / 1000 for i in gtsrb_running_time],
        [i / 1000 for i in tinyimagenet_running_time],
    )

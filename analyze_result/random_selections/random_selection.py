from random import randrange

selection_counts = [  # MNSIT 1,1-2,6 EMNIST 1,1-2,4
    96,
    720,
    3360,
    10920,
    26208,
    48048,
    1800,
    8400,
    27300,
    65520,
    120120,
    200,
    1900,
    11400,
    48450,
    8550,
    51300,
    218025,
]
greedy_rankings = [1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 7, 1, 1, 1, 3, 1, 1, 6]
greedy_time_consuming = [
    22,
    37,
    52,
    65,
    78,
    89,
    42,
    55,
    68,
    80,
    91,
    57,
    94,
    132,
    170,
    104,
    142,
    176,
]

single_random_time_consuming = 2
random_repeat_times = 1000

# average ranking
average_best_ranking = []
for i, count in enumerate(selection_counts):
    best_rank_sum = 0
    for _ in range(random_repeat_times):
        best_rank = 2**32
        for j in range(
            int(greedy_time_consuming[i] / single_random_time_consuming) + 1
        ):
            rank = randrange(1, count + 1)
            best_rank = min(rank, best_rank)
        best_rank_sum += best_rank
    average_best_ranking.append(best_rank_sum / random_repeat_times)

print(f"average best ranking: {average_best_ranking}")

# average time consuming
average_min_time_consuming = []
for i, count in enumerate(selection_counts):
    time_sum = 0
    for _ in range(random_repeat_times):
        time = 0
        rank = randrange(1, count + 1)
        while rank > greedy_rankings[i]:
            time += single_random_time_consuming
            rank = randrange(1, count + 1)
        time_sum += time
    average_min_time_consuming.append(time_sum / random_repeat_times)

print(f"average min time consuming: {average_min_time_consuming}")

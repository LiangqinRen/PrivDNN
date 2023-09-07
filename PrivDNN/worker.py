import data
import utils
import models

import re
import torch
import copy
import json
import math
import random
import sys
import threading

import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
import itertools as it
import numpy as np

from torch.ao.pruning._experimental.pruner.FPGM_pruner import FPGMPruner


def save_model(trained_model, model_save_path):
    torch.save(trained_model, model_save_path)


def load_trained_model(model_path):
    model = torch.load(model_path)
    model.eval()  # use the evaluation mode

    return model


def save_selected_neurons(dataloaders, selected_neurons, file_name=None):
    path = f"../saved_models/{dataloaders['name']}/"
    path += file_name if file_name else "selected_neurons.json"

    with open(path, "w") as convert_file:
        convert_file.write(json.dumps(selected_neurons))


def load_selected_neurons(dataloaders, file_name=None):
    path = f"../saved_models/{dataloaders['name']}/"
    path += file_name if file_name else "selected_neurons.json"

    selected_neurons = {}
    with open(path, "r") as convert_file:
        selected_neurons = json.load(convert_file)

    int_keys_selected_neurons = {}
    for key, value in selected_neurons.items():
        int_keys_selected_neurons[int(key)] = value

    return int_keys_selected_neurons


def get_model(logger, dataloaders, model_path):
    if utils.is_file_exist(model_path):
        logger.info("use the existed model")
        return load_trained_model(model_path)

    model_list = {
        "MNIST": models.SplitMNISTNet(),
        "EMNIST": models.SplitEMNISTNet(),
        "GTSRB": models.SplitGTSRBNet(),
        "CIFAR10": models.SplitCIFAR10Net(),
    }
    model = model_list[dataloaders["name"]]
    model.set_layers_on_cuda()
    logger.info("use the new model")

    return model


def get_model_accuracy(model, dataloader):
    correct_count = 0
    samples_count = len(dataloader.dataset)
    label_correct_count = {}

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.cuda()
            labels = labels.cuda()
            scores = model(imgs)
            _, predictions = scores.max(1)
            correct_count += (predictions == labels).sum()

            for i in range(len(predictions)):
                if not labels[i].item() in label_correct_count:
                    label_correct_count[labels[i].item()] = [0, 0]

                label_correct_count[labels[i].item()][1] += 1

                if predictions[i] == labels[i]:
                    label_correct_count[labels[i].item()][0] += 1

    accuracy = float(f"{float(correct_count) / float(samples_count) * 100:.2f}")

    return correct_count, samples_count, accuracy, label_correct_count


def test_model(logger, model, dataloaders):
    correct_count, samples_count, accuracy, _ = get_model_accuracy(
        model, dataloaders["test"]
    )
    logger.info(f"[{correct_count}/{samples_count}], Accuracy: {accuracy:.2f}%")


def closing_test(args, logger, model, dataloaders, selected_neurons, file_name=None):
    (
        separate_accuracy,
        separate_label_correct_count,
    ) = get_accuracy_after_separating_neurons(
        copy.deepcopy(model), dataloaders, selected_neurons
    )
    remove_accuracy, remove_label_correct_count = get_accuracy_after_removing_neurons(
        copy.deepcopy(model), dataloaders, selected_neurons
    )

    point = get_neuron_point(args, separate_accuracy, remove_accuracy)

    logger.info(
        f"{dataloaders['name']} selecte neurons {selected_neurons}, we get the accuracy [{separate_accuracy}% - {remove_accuracy}% = {separate_accuracy - remove_accuracy:.2f}%, point {point:.2f}]"
    )

    _, _, _, original_label_correct_count = get_model_accuracy(
        model, dataloaders["test"]
    )
    original_label_info = "original label accuracy:\n"
    for label, count in sorted(original_label_correct_count.items()):
        original_label_info += (
            f"{label:3}: {count[0]:4}/{count[1]:4} {count[0]/count[1]*100:.2f}\n"
        )
    logger.info(original_label_info)

    separate_label_info = "separate label accuracy:\n"
    separate_correct_count = []
    for label, count in sorted(separate_label_correct_count.items()):
        separate_correct_count.append((label, count[0], count[1]))
        separate_label_info += (
            f"{label:3}: {count[0]:4}/{count[1]:4} {count[0]/count[1]*100:.2f}\n"
        )
    logger.info(separate_label_info)

    remove_label_info = "remove label accuracy:\n"
    remove_correct_count = []
    for label, count in sorted(remove_label_correct_count.items()):
        remove_correct_count.append((label, count[0], count[1]))
        remove_label_info += (
            f"{label:3}: {count[0]:4}/{count[1]:4} {count[0]/count[1]*100:.2f}\n"
        )
    logger.info(remove_label_info)

    worse_performance_labels = []
    for i, j in zip(separate_correct_count, remove_correct_count):
        if i[1] < j[1]:
            worse_performance_labels.append(i[0])

    logger.info(f"worse performance label: {worse_performance_labels}")

    save_selected_neurons(dataloaders, selected_neurons, file_name)


def train_model(args, logger, model, dataloaders, parameters, model_path=None):
    # if model_path is not none, save the best model every time
    criterion = nn.CrossEntropyLoss()
    optimizer = scheduler = None

    if dataloaders["name"] == "MNIST":
        if isinstance(parameters[0], list):
            selected_parameters = parameters[0]
            others_parameters = parameters[1]
            optimizer = optim.Adam(
                [
                    {"params": selected_parameters, "lr": 7.5e-4},
                    {"params": others_parameters, "lr": 7.5e-5},
                ]
            )
        else:
            optimizer = optim.Adam(parameters, lr=7.5e-3)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=dataloaders["epoch"]
        )
    elif dataloaders["name"] == "EMNIST":
        if isinstance(parameters[0], list):
            selected_parameters = parameters[0]
            others_parameters = parameters[1]
            optimizer = optim.Adam(
                [
                    {"params": selected_parameters, "lr": 1e-4},
                    {"params": others_parameters, "lr": 1e-5},
                ]
            )
        else:
            optimizer = optim.Adam(parameters, lr=1e-3)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=dataloaders["epoch"]
        )
    elif dataloaders["name"] == "GTSRB":
        if isinstance(parameters[0], list):
            selected_parameters = parameters[0]
            others_parameters = parameters[1]
            optimizer = optim.SGD(
                [
                    {"params": selected_parameters, "lr": 5e-3},
                    {"params": others_parameters, "lr": 5e-4},
                ],
                momentum=0.9,
                weight_decay=5e-4,
            )
        else:
            optimizer = optim.SGD(parameters, lr=5e-2, momentum=0.9)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=dataloaders["epoch"]
        )
    elif dataloaders["name"] == "CIFAR10":
        if isinstance(parameters[0], list):
            selected_parameters = parameters[0]
            others_parameters = parameters[1]
            optimizer = optim.SGD(
                [
                    {"params": selected_parameters, "lr": 5e-3},
                    {"params": others_parameters, "lr": 5e-4},
                ],
                momentum=0.9,
                weight_decay=5e-4,
            )
        else:
            optimizer = optim.SGD(parameters, lr=5e-2, momentum=0.9)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=dataloaders["epoch"]
        )

    best_loss = 1
    best_model = None

    for i in range(dataloaders["epoch"]):
        loss_ep = 0
        for imgs, labels in dataloaders["train"]:
            imgs = imgs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            scores = model(imgs)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()

            loss_ep += loss.item()

        average_loss = loss_ep / len(dataloaders["train"].dataset)
        if average_loss < best_loss:
            best_loss = average_loss
            best_model = copy.deepcopy(model)

        scheduler.step()
        with torch.no_grad():
            _, _, accuracy, _ = get_model_accuracy(model, dataloaders["validate"])
            logger.info(
                f"[Epoch {i:3}]Loss: {average_loss:.8f}, Accuracy: {accuracy:5.3f}%"
            )

        if model_path and best_model:
            if args.model_work_mode == utils.ModelWorkMode.train:
                best_model.copy_parameters_to_split_model()

            if args.model_work_mode == utils.ModelWorkMode.recover:
                save_model(model, model_path)
            else:
                save_model(best_model, model_path)

        # get more data, partial label, more data, test
    if args.model_work_mode != utils.ModelWorkMode.recover:
        model = best_model


def train_and_save_model(args, logger, dataloaders, model_path):
    model = get_model(logger, dataloaders, model_path)
    logger.info("train the model")

    layers_list = model.get_layers_list(include_fc_layers=True)
    parameters = []
    for layers in layers_list:
        if isinstance(layers, nn.Linear) or isinstance(layers, nn.BatchNorm2d):
            parameters.extend(list(layers.parameters()))
        else:
            parameters.extend(list(layers[0].parameters()))

    train_model(args, logger, model, dataloaders, parameters, model_path)


def train_and_save_percent_dataset_model(args, logger, dataloaders, percent_range):
    accuracies = []
    for percent in range(
        percent_range[0], percent_range[1] + percent_range[2], percent_range[2]
    ):
        logger.info(f"train the model with {percent}% data")

        percent_dataloaders = copy.deepcopy(dataloaders)
        dataloaders.use_partial_dataloaders(percent_dataloaders, percent=percent)

        percent_model_path = utils.get_model_path(args, percent_dataloaders, percent)
        train_and_save_model(args, logger, percent_dataloaders, percent_model_path)
        trained_model = load_trained_model(percent_model_path)

        _, _, accuracy, _ = get_model_accuracy(
            trained_model, percent_dataloaders["test"]
        )
        accuracies.append(accuracy)
        logger.info(
            f"[{len(percent_dataloaders['train'].dataset)}, {percent}%]Accuracy: {accuracy}%"
        )

    logger.info(
        f"{percent_range[0]}% - {percent_range[1]}% with interval {percent_range[2]}% train dataset get accuracies: {accuracies}"
    )


def get_accuracy_after_removing_neurons(model, dataloaders, selected_neurons):
    layers_list = model.get_layers_list()
    for i in list(selected_neurons.keys()):
        layers_list[i - 1][0].set_neurons_to_remove(selected_neurons[i])

    _, _, accuracy, label_correct_count = get_model_accuracy(model, dataloaders["test"])

    return accuracy, label_correct_count


def get_first_layer_neurons_to_separate(
    args,
    logger,
    model,
    first_layer_index,
    dataloaders,
    neurons_limit,
    neurons_pool=None,
    pre_select=[],
):
    layers_list = model.get_layers_list()
    first_layer = layers_list[first_layer_index][0]
    total_pool = neurons_pool if neurons_pool else range(first_layer.layer.out_channels)

    selected_layer_neurons = pre_select
    while len(selected_layer_neurons) < neurons_limit:
        accuracies = []
        for neuron_index in [i for i in total_pool if i not in selected_layer_neurons]:
            layer_neuron_to_remove = {
                first_layer_index + 1: selected_layer_neurons + [neuron_index]
            }
            accuracy, _ = get_accuracy_after_removing_neurons(
                copy.deepcopy(model), dataloaders, layer_neuron_to_remove
            )
            accuracies.append([accuracy, neuron_index])

        accuracies.sort(reverse=True)
        for neuron in accuracies[: args.greedy_step]:
            selected_layer_neurons.append(neuron[1])
        logger.info(
            f"conv {first_layer.layer_index} sorted neuron accuracies: {accuracies}, selected_neurons: {selected_layer_neurons}"
        )

    return selected_layer_neurons[:neurons_limit]


def get_accuracy_after_separating_neurons(model, dataloaders, selected_neurons):
    layers_list = model.get_layers_list()
    for i in list(selected_neurons.keys())[1:]:
        current_layer = layers_list[i - 1][0]
        current_layer.set_neurons_to_separate(
            selected_neurons[i - 1], selected_neurons[i]
        )

    _, _, accuracy, label_correct_count = get_model_accuracy(model, dataloaders["test"])

    return accuracy, label_correct_count


def get_layer_neurons_to_separate(
    args,
    logger,
    model,
    dataloaders,
    selected_neurons,
    select_limit,
    index,
    neurons_pool=None,
):
    layers_list = model.get_layers_list()
    selected_layer_neurons = (
        selected_neurons[index + 1] if index + 1 in selected_neurons else []
    )

    total_pool = (
        neurons_pool
        if neurons_pool
        else [*range(layers_list[index][0].layer.out_channels)]
    )

    while len(selected_layer_neurons) < select_limit:
        accuracies = []
        for neuron_index in [i for i in total_pool if i not in selected_layer_neurons]:
            index_selected_neurons = copy.deepcopy(selected_neurons)
            index_selected_neurons[index + 1] = selected_layer_neurons + [neuron_index]

            separating_accuracy, _ = get_accuracy_after_separating_neurons(
                copy.deepcopy(model), dataloaders, copy.deepcopy(index_selected_neurons)
            )
            removing_accuracy, _ = get_accuracy_after_removing_neurons(
                copy.deepcopy(model), dataloaders, copy.deepcopy(index_selected_neurons)
            )

            accuracies.append(
                (
                    get_neuron_point(args, separating_accuracy, removing_accuracy),
                    neuron_index,
                    [separating_accuracy, removing_accuracy],
                )
            )

        accuracies.sort(reverse=True)
        for neuron in accuracies[: args.greedy_step]:
            selected_layer_neurons.append(neuron[1])

        logger.info(
            f"conv {layers_list[index][0].layer_index} sorted neuron accuracies: {accuracies}, selected_neurons: {selected_layer_neurons}"
        )

    return selected_layer_neurons


def select_full_combination_thread(
    args, logger, model, dataloaders, combinations, file_name
):
    results = []  # combination, separating accuracy, removing accuracy
    for index, selected_neurons in enumerate(combinations):
        separate_accuracy, _ = get_accuracy_after_separating_neurons(
            copy.deepcopy(model),
            dataloaders,
            selected_neurons,
        )
        remove_accuracy, _ = get_accuracy_after_removing_neurons(
            copy.deepcopy(model),
            dataloaders,
            selected_neurons,
        )
        logger.info(
            f"{index:<6} :{selected_neurons}[{separate_accuracy:.2f}% - {remove_accuracy:.2f}% = {separate_accuracy - remove_accuracy:.2f}%]"
        )

        results.append([index, selected_neurons, separate_accuracy, remove_accuracy])

    results = np.array(results)
    np.save(file_name, results, allow_pickle=True)


def select_full_combination(args, logger, model, dataloaders):
    if dataloaders["name"] != "MNIST" and dataloaders["name"] != "EMNIST":
        logger.error(
            f"the dataset is {dataloaders['name']}, while this function only supports MNIST and EMNIST"
        )
        quit()

    combinations = []
    for i in range(1, 1 + 1):
        for layer_1_neurons in it.combinations(range(0, 10), i):
            for j in range(4, 4 + 1):
                for layer_2_neurons in it.combinations(range(0, 20), j):
                    selected_neurons = {1: layer_1_neurons, 2: layer_2_neurons}
                    combinations.append(selected_neurons)

    thread_count = 1
    index = int(len(combinations) / thread_count) + len(combinations) % thread_count
    combinations_thread = [combinations[0:index]]

    for i in range(1, thread_count):
        combinations_thread.append(
            combinations[index : index + int(len(combinations) / thread_count)]
        )
        index += int(len(combinations) / thread_count)

    threads = []
    for i in range(thread_count):
        worker = threading.Thread(
            target=select_full_combination_thread,
            args=(
                args,
                logger,
                model,
                dataloaders,
                combinations_thread[i],
                f"thread_{i}.npy",
            ),
        )
        threads.append(worker)
        worker.start()

    for worker in threads:
        worker.join()


def select_neurons_v1(args, logger, model, dataloaders):
    # random
    layers_list = model.get_layers_list()
    select_limit = args.initial_layer_neurons
    first_layer_index = args.initial_layer_index  # start from 0
    encrypt_layers_count = args.encrypt_layers_count

    selected_neurons = {}
    for i in range(first_layer_index):
        selected_neurons[i + 1] = []

    for i in range(first_layer_index, first_layer_index + encrypt_layers_count):
        if args.percent_factor is not None:
            select_limit = max(
                1,
                int(layers_list[i][0].layer.out_channels * args.percent_factor / 100),
            )

        selected_neurons[i + 1] = random.choices(
            range(layers_list[i][0].layer.out_channels), k=select_limit
        )

        if args.add_factor is not None:
            select_limit = select_limit + args.add_factor
        elif args.multiply_factor is not None:
            select_limit = select_limit * args.multiply_factor

    closing_test(args, logger, model, dataloaders, selected_neurons)


def sig(x):
    return 1 / (1 + math.exp(-x))


def get_neuron_point(args, separating_accuracy, removing_accuracy):
    accuracy = args.accuracy_base

    point = -sig((accuracy - separating_accuracy) / 1) + sig(
        (separating_accuracy - removing_accuracy) / 5
    )

    return point


"""def get_neuron_point(args, separating_accuracy, removing_accuracy, no_penalty=True):
    point = 0
    if no_penalty:
        point = args.alpha * (separating_accuracy - args.accuracy_base) + args.beta * (
            separating_accuracy - removing_accuracy
        )
    else:
        separating_accuracy_factor = (
            1 if separating_accuracy > args.accuracy_base else 10 * args.alpha
        )
        difference_factor = (
            math.log(
                separating_accuracy - removing_accuracy,
                args.difference_base,
            )
            if separating_accuracy > removing_accuracy
            else -10 * args.beta
        )

        point += (
            args.alpha
            * (separating_accuracy - args.accuracy_base)
            * separating_accuracy_factor
        )
        point += args.beta * args.difference_base * difference_factor

    return round(point, 2)"""


def select_neurons_v1_multi(args, logger, model, dataloaders):
    # random
    layers_list = model.get_layers_list()
    first_layer_index = args.initial_layer_index  # start from 0
    encrypt_layers_count = args.encrypt_layers_count

    selections = {}
    for i in range(args.random_selection_times):
        selected_neurons = {}
        for i in range(first_layer_index):
            selected_neurons[i + 1] = []

        select_limit = args.initial_layer_neurons
        for i in range(first_layer_index, first_layer_index + encrypt_layers_count):
            if args.percent_factor is not None:
                select_limit = max(
                    1,
                    int(
                        layers_list[i][0].layer.out_channels * args.percent_factor / 100
                    ),
                )

            selected_neurons[i + 1] = random.sample(
                range(layers_list[i][0].layer.out_channels), select_limit
            )

            if args.add_factor is not None:
                select_limit = select_limit + args.add_factor
            elif args.multiply_factor is not None:
                select_limit = select_limit * args.multiply_factor

        separate_accuracy, _ = get_accuracy_after_separating_neurons(
            copy.deepcopy(model), dataloaders, copy.deepcopy(selected_neurons)
        )
        remove_accuracy, _ = get_accuracy_after_removing_neurons(
            copy.deepcopy(model), dataloaders, copy.deepcopy(selected_neurons)
        )
        logger.info(
            f"{dataloaders['name']}{selected_neurons}[{separate_accuracy:.2f}% - {remove_accuracy:.2f}% = {separate_accuracy - remove_accuracy:.2f}%]"
        )
        selections[json.dumps(selected_neurons)] = [separate_accuracy, remove_accuracy]

    points = []
    separate_accuracies = []
    remove_accuracies = []
    for selection in selections.items():
        # combination format: {{combination}:[separating_accuracy, removing_accuracy]}
        selected_neurons = selection[0]
        accuracies = selection[1]
        separating_accuracy = float(selection[1][0])
        removing_accuracy = float(selection[1][1])

        separate_accuracies.append(separating_accuracy)
        remove_accuracies.append(removing_accuracy)

        points.append(
            [
                get_neuron_point(args, separating_accuracy, removing_accuracy),
                accuracies,
                selected_neurons,
            ]
        )

    average_separate_accuracy = sum(separate_accuracies) / len(separate_accuracies)
    average_remove_accuracy = sum(remove_accuracies) / len(remove_accuracies)
    points.sort(reverse=True)

    for i in range(3):
        logger.info(
            f"{dataloaders['name']} max point {points[i][0]:.3f} of {args.random_selection_times} times with selected neurons {points[i][2]}:[{points[i][1][0]:.2f}% - {points[i][1][1]:.2f}% = {points[i][1][0] - points[i][1][1]:.2f}%]"
        )
    for i in range(3):
        logger.info(
            f"{dataloaders['name']} min point {points[-1-i][0]:.3f} of {args.random_selection_times} times with selected neurons {points[-1-i][2]}:[{points[-1-i][1][0]:.2f}% - {points[-1-i][1][1]:.2f}% = {points[-1-i][1][0] - points[-1-i][1][1]:.2f}%]"
        )

    logger.info(
        f"{dataloaders['name']} average resule of {args.random_selection_times} times:[{average_separate_accuracy:.2f}% - {average_remove_accuracy:.2f}% = {average_separate_accuracy - average_remove_accuracy:.2f}%]"
    )


def select_neurons_v2(args, logger, model, dataloaders):
    # greedy
    layers_list = model.get_layers_list()
    first_layer_neurons_count = args.initial_layer_neurons
    first_layer_index = args.initial_layer_index  # start from 0
    encrypt_layers_count = args.encrypt_layers_count
    if args.percent_factor is not None:
        first_layer_neurons_count = max(
            1,
            int(
                layers_list[args.initial_layer_index][0].layer.out_channels
                * args.percent_factor
                / 100
            ),
        )

    selected_neurons = {}
    for i in range(first_layer_index):
        selected_neurons[i + 1] = []

    selected_first_layer_neurons = get_first_layer_neurons_to_separate(
        args,
        logger,
        copy.deepcopy(model),
        first_layer_index,
        dataloaders,
        first_layer_neurons_count,
    )
    selected_neurons[first_layer_index + 1] = selected_first_layer_neurons

    logger.info(
        f"[conv{layers_list[first_layer_index][0].layer_index}]first layer selected neurons list: {selected_first_layer_neurons}"
    )

    for i in range(first_layer_index + 1, first_layer_index + encrypt_layers_count):
        last_layer_selected_count = len(selected_neurons[i])
        select_limit = 0
        if args.percent_factor is not None:
            select_limit = max(
                1, int(layers_list[i][0].layer.out_channels * args.percent_factor / 100)
            )
        elif args.add_factor is not None:
            select_limit = last_layer_selected_count + args.add_factor
        else:
            select_limit = last_layer_selected_count * args.multiply_factor

        selected_layer_neurons = get_layer_neurons_to_separate(
            args,
            logger,
            copy.deepcopy(model),
            dataloaders,
            copy.deepcopy(selected_neurons),
            select_limit,
            i,
        )

        selected_neurons[i + 1] = selected_layer_neurons

        logger.info(
            f"[conv{layers_list[i][0].layer_index}]Seperated neurons list: {selected_neurons}"
        )

        if len(selected_neurons[i + 1]) == layers_list[i][0].layer.out_channels:
            while i + 1 < len(layers_list):
                i += 1
                selected_neurons[i + 1] = range(layers_list[i][0].layer.out_channels)
            break

    closing_test(args, logger, copy.deepcopy(model), dataloaders, selected_neurons)


def select_neurons_v2_amend(args, logger, model, dataloaders, input_file, output_file):
    # only for large models
    selected_neurons = load_selected_neurons(dataloaders, input_file)

    layers_list = model.get_layers_list()
    first_layer_neurons_count = args.initial_layer_neurons
    first_layer_index = args.initial_layer_index  # start from 0
    encrypt_layers_count = args.encrypt_layers_count
    if args.percent_factor is not None:
        first_layer_neurons_count = max(
            1,
            int(
                layers_list[args.initial_layer_index][0].layer.out_channels
                * args.percent_factor
                / 100
            ),
        )

    first_layer_filters_pool = []
    for i in range(layers_list[first_layer_index][0].layer.out_channels):
        if i not in selected_neurons[first_layer_index + 1]:
            first_layer_filters_pool.append(i)

    selected_first_layer_neurons = get_first_layer_neurons_to_separate(
        args,
        logger,
        copy.deepcopy(model),
        first_layer_index,
        dataloaders,
        first_layer_neurons_count,
        neurons_pool=first_layer_filters_pool,
        pre_select=copy.deepcopy(selected_neurons[first_layer_index + 1]),
    )

    selected_neurons[first_layer_index + 1] = selected_first_layer_neurons

    logger.info(
        f"[conv{layers_list[first_layer_index][0].layer_index}]first layer selected neurons list: {selected_first_layer_neurons}"
    )

    for i in range(first_layer_index + 1, first_layer_index + encrypt_layers_count):
        select_limit = max(
            1, int(layers_list[i][0].layer.out_channels * args.percent_factor / 100)
        )

        selected_layer_neurons = get_layer_neurons_to_separate(
            args,
            logger,
            copy.deepcopy(model),
            dataloaders,
            copy.deepcopy(selected_neurons),
            select_limit,
            i,
        )

        selected_neurons[i + 1] = selected_layer_neurons

        logger.info(
            f"[conv{layers_list[i][0].layer_index}]Seperated neurons list: {selected_neurons}"
        )

        if len(selected_neurons[i + 1]) == layers_list[i][0].layer.out_channels:
            while i + 1 < len(layers_list):
                i += 1
                selected_neurons[i + 1] = range(layers_list[i][0].layer.out_channels)
            break

    closing_test(
        args, logger, copy.deepcopy(model), dataloaders, selected_neurons, output_file
    )


def pruning_select_norm(model, index, select_limit):
    layers = model.get_layers_list()[index]
    prune_count = layers[0].layer.out_channels - select_limit
    prune.ln_structured(layers[0].layer, name="weight", amount=prune_count, n=2, dim=0)

    selected_layer_neurons = []
    for i, weight in enumerate(layers[0].layer.weight):
        if abs(torch.sum(weight)) > 0.000001:
            selected_layer_neurons.append(i)

    # selected_layer_neurons.sort()
    return selected_layer_neurons


def pruning_select_fpgm(model, index, select_limit):
    pruner = FPGMPruner()
    layers = model.get_layers_list()[index]
    layer_weight = layers[0].layer.weight.data
    distance = pruner._compute_distance(layer_weight)
    indexed_distance = [(j.item(), i) for i, j in enumerate(distance)]
    indexed_distance.sort(reverse=True)

    selected_layer_neurons = []
    for i in range(select_limit):
        selected_layer_neurons.append(indexed_distance[i][1])

    # selected_layer_neurons.sort()
    return selected_layer_neurons


def get_feature_hook(self, input, output):
    global feature_result
    global total
    a = output.shape[0]
    b = output.shape[1]
    c = torch.tensor(
        [
            torch.linalg.matrix_rank(output[i, j, :, :]).item()
            for i in range(a)
            for j in range(b)
        ]
    )

    c = c.view(a, -1).float()
    c = c.sum(0)
    feature_result = feature_result * total + c
    total = total + a
    feature_result = feature_result / total


feature_result = torch.tensor(0.0)
total = torch.tensor(0.0)


def pruning_select_hrank(model, index, select_limit, dataloaders):
    global feature_result
    global total

    modified_model = copy.deepcopy(model)
    layers = modified_model.get_layers_list()[index]

    dataloaders_500_train = copy.deepcopy(dataloaders)
    data.use_partial_dataloaders(dataloaders_500_train, count=500)

    handler = layers[0].register_forward_hook(get_feature_hook)
    _, _, _, _ = get_model_accuracy(modified_model, dataloaders["train"])
    handler.remove()

    ranks = []
    for i, rank in enumerate(feature_result.tolist()):
        ranks.append([rank, i])
    ranks.sort(reverse=True)

    feature_result = torch.tensor(0.0)
    total = torch.tensor(0.0)

    selected_layer_neurons = []
    for i in range(select_limit):
        selected_layer_neurons.append(ranks[i][1])

    # selected_layer_neurons.sort()
    return selected_layer_neurons


def pruning_select_greedy_forward(
    model, index, select_limit, dataloaders, selected_neurons
):
    # the function has `CUDA out of memory` error when select complex datasets

    # reset previous layers parameters
    reset_model = copy.deepcopy(model)
    reset_model.work_mode = models.WorkMode.split
    for key, value in selected_neurons.items():
        layers = reset_model.get_layers_list()[key - 1]
        for neuron_index in [
            i for i in range(layers[0].layer.out_channels) if i not in value
        ]:
            layers[neuron_index + 1].layer.reset_parameters()

    dataloaders_512_train = copy.deepcopy(dataloaders)
    data.use_partial_dataloaders(dataloaders_512_train, count=512)

    # reset current layer parameters
    current_reset_model = copy.deepcopy(reset_model)
    for i in range(current_reset_model.get_layers_list()[index][0].layer.out_channels):
        current_reset_model.get_layers_list()[index][i + 1].layer.reset_parameters()

    original_loss = 0
    criterion = nn.CrossEntropyLoss()
    for imgs, labels in dataloaders_512_train["train"]:
        imgs = imgs.cuda()
        labels = labels.cuda()

        scores = current_reset_model(imgs)
        original_loss += criterion(scores, labels)
    original_loss = original_loss.item()

    # select neurons
    selected_layer_neurons = []
    layers = reset_model.get_layers_list()[index]
    for _ in range(select_limit):
        loss_rank = []
        for neuron_index in [
            i
            for i in range(layers[0].layer.out_channels)
            if i not in selected_layer_neurons
        ]:
            iter_model = copy.deepcopy(reset_model)
            for j in range(iter_model.get_layers_list()[index][0].layer.out_channels):
                if j != i:
                    iter_model.get_layers_list()[index][j + 1].layer.reset_parameters()

            current_loss = 0
            for imgs, labels in dataloaders["train"]:
                imgs = imgs.cuda()
                labels = labels.cuda()

                scores = model(imgs)
                current_loss += criterion(scores, labels)
            current_loss = current_loss.item()

            loss_rank.append([original_loss - current_loss, neuron_index])

        loss_rank.sort(reverse=True)
        selected_layer_neurons.append(loss_rank[0][1])

    # selected_layer_neurons.sort()
    return selected_layer_neurons


def select_neurons_v3(args, logger, model, dataloaders, prune_index=1):
    # pruning
    layers_list = model.get_layers_list()
    select_limit = args.initial_layer_neurons
    first_layer_index = args.initial_layer_index  # start from 0
    encrypt_layers_count = args.encrypt_layers_count

    selected_neurons = {}
    for i in range(first_layer_index):
        selected_neurons[i + 1] = []

    for i in range(first_layer_index, first_layer_index + encrypt_layers_count):
        if args.percent_factor is not None:
            select_limit = max(
                1,
                int(layers_list[i][0].layer.out_channels * args.percent_factor / 100),
            )

        selected_neurons[i + 1] = []
        if prune_index == 1:
            selected_neurons[i + 1] = pruning_select_norm(
                copy.deepcopy(model), i, select_limit
            )
        elif prune_index == 2:
            selected_neurons[i + 1] = pruning_select_fpgm(
                copy.deepcopy(model), i, select_limit
            )
        elif prune_index == 3:
            selected_neurons[i + 1] = pruning_select_hrank(
                copy.deepcopy(model), i, select_limit, dataloaders
            )
        elif prune_index == 4:
            selected_neurons[i + 1] = pruning_select_greedy_forward(
                copy.deepcopy(model),
                i,
                select_limit,
                dataloaders,
                copy.deepcopy(selected_neurons),
            )
        else:
            logger.fatal(f"unsuppoted prune algorithm index {prune_index}")
            exit()

        if args.add_factor is not None:
            select_limit = select_limit + args.add_factor
        elif args.multiply_factor is not None:
            select_limit = select_limit * args.multiply_factor

    closing_test(args, logger, copy.deepcopy(model), dataloaders, selected_neurons)


def select_neurons_v4(args, logger, model, dataloaders, prune_index=1):
    # pruning + greedy
    layers_list = model.get_layers_list()
    first_layer_neurons_count = args.initial_layer_neurons
    first_layer_index = args.initial_layer_index  # start from 0
    encrypt_layers_count = args.encrypt_layers_count
    if args.percent_factor is not None:
        first_layer_neurons_count = max(
            1,
            int(
                layers_list[args.initial_layer_index][0].layer.out_channels
                * args.percent_factor
                / 100
            ),
        )

    selected_neurons = {}
    for i in range(first_layer_index):
        selected_neurons[i + 1] = []

    first_layer_pool = pruning_select_norm(
        copy.deepcopy(model), 0, first_layer_neurons_count * 2
    )
    selected_first_layer_neurons = get_first_layer_neurons_to_separate(
        args,
        logger,
        copy.deepcopy(model),
        first_layer_index,
        dataloaders,
        first_layer_neurons_count,
        neurons_pool=first_layer_pool,
    )
    selected_neurons[first_layer_index + 1] = selected_first_layer_neurons

    logger.info(
        f"[conv{layers_list[first_layer_index][0].layer_index}]first layer selected neurons list: {selected_first_layer_neurons}"
    )

    for i in range(first_layer_index + 1, first_layer_index + encrypt_layers_count):
        last_layer_selected_count = len(selected_neurons[i])
        select_limit = 0
        if args.percent_factor is not None:
            select_limit = max(
                1, int(layers_list[i][0].layer.out_channels * args.percent_factor / 100)
            )
        elif args.add_factor is not None:
            select_limit = last_layer_selected_count + args.add_factor
        else:
            select_limit = last_layer_selected_count * args.multiply_factor

        layer_pool = None
        if prune_index == 1:
            layer_pool = pruning_select_norm(copy.deepcopy(model), i, select_limit * 2)
        elif prune_index == 2:
            layer_pool = pruning_select_fpgm(copy.deepcopy(model), i, select_limit * 2)

        selected_layer_neurons = get_layer_neurons_to_separate(
            args,
            logger,
            copy.deepcopy(model),
            dataloaders,
            copy.deepcopy(selected_neurons),
            select_limit,
            i,
            layer_pool,
        )

        selected_neurons[i + 1] = selected_layer_neurons

        logger.info(
            f"[conv{layers_list[i][0].layer_index}]Seperated neurons list: {selected_neurons}"
        )

        if len(selected_neurons[i + 1]) == layers_list[i][0].layer.out_channels:
            while i + 1 < len(layers_list):
                i += 1
                selected_neurons[i + 1] = range(layers_list[i][0].layer.out_channels)
            break

    closing_test(args, logger, copy.deepcopy(model), dataloaders, selected_neurons)


def recover_model(args, logger, model, dataloaders, model_path):
    dataloaders_recover = copy.deepcopy(dataloaders)
    dataloaders_recover["epoch"] = 64
    if args.recover_dataset_percent:
        data.use_partial_dataloaders(
            dataloaders_recover, percent=args.recover_dataset_percent, mode="test"
        )
    elif args.recover_dataset_count:
        data.use_partial_dataloaders(
            dataloaders_recover, count=args.recover_dataset_count, mode="test"
        )

    recover_model_path = (
        model_path[:-7] + str(args.recover_dataset_percent) + "_recover.pth"
    )

    selected_neurons = load_selected_neurons(
        dataloaders,
        f"selected_neurons_{int(args.percent_factor)}%.json"
        if args.percent_factor
        else f"recover_selected_neurons.json",
    )
    model.selected_neurons = selected_neurons

    logger.info("original accuracy")
    model.work_mode = models.WorkMode.split
    test_model(logger, model, dataloaders)

    recover_parameters = []
    others_parameters = []
    for layers in model.get_layers_list(True):
        if isinstance(layers, nn.Linear) or isinstance(layers, nn.BatchNorm2d):
            others_parameters.extend(list(layers.parameters()))
        else:
            layer_index = layers[0].layer_index  # layer_index starts from 1
            others_parameters.extend(list(layers[0].parameters()))
            for i in range(1, len(layers)):
                if (
                    layer_index in selected_neurons
                    and i - 1 in selected_neurons[layer_index]
                ):
                    layers[i].layer.reset_parameters()
                    recover_parameters.extend(list(layers[i].parameters()))
                else:
                    others_parameters.extend(list(layers[i].parameters()))

    train_model(
        args,
        logger,
        model,
        dataloaders_recover,
        [recover_parameters, []],
        recover_model_path,
    )

    logger.info("after recovering the model(ReLU)")
    test_model(logger, model, dataloaders)


def test_separated_model(args, logger, model, dataloaders):
    selected_neurons = load_selected_neurons(dataloaders, args.selected_neurons_file)
    closing_test(args, logger, model, dataloaders, selected_neurons)
import data
import utils
import models

import torch
import copy
import json
import math
import threading
import inspect
import random

import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
import itertools as it
import numpy as np
import matplotlib.pyplot as plt

from torch.nn.modules.container import Sequential
from torchvision.utils import save_image
from torch.ao.pruning._experimental.pruner.FPGM_pruner import FPGMPruner
from tqdm import tqdm


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
        "TinyImageNet": models.SplitTinyImageNet(),
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
            _, predictions = torch.topk(scores, 5, dim=1)
            # print(predictions.shape)
            # quit()
            """print(scores[0])
            _, predictions = scores[0].max(1)
            print(_, predictions)
            quit()
            _, predictions = scores.max(1)
            print(_, predictions)
            quit()"""
            # _, predictions = scores.max(1)
            pre = torch.split(predictions, 1, dim=1)
            """print(pre[0].view(-1).shape)
            _, predictions = scores.max(1)
            print(predictions.shape)
            print(pre[0].view(-1) - predictions)
            quit()"""

            correct_count += (pre[0].view(-1) == labels).sum()
            correct_count += (pre[1].view(-1) == labels).sum()
            correct_count += (pre[2].view(-1) == labels).sum()
            correct_count += (pre[3].view(-1) == labels).sum()
            correct_count += (pre[4].view(-1) == labels).sum()

            # correct_count = torch.eq(predictions[:, None, ...], labels).any(dim=1)
            # take the mean of correct_pixels to get the overall average top-k accuracy:
            # top_k_acc = correct_pixels.mean()

            """for i in range(len(predictions)):
                if not labels[i].item() in label_correct_count:
                    label_correct_count[labels[i].item()] = [0, 0]

                label_correct_count[labels[i].item()][1] += 1

                if predictions[i] == labels[i]:
                    label_correct_count[labels[i].item()][0] += 1"""

    accuracy = float(f"{float(correct_count) / float(samples_count) * 100:.2f}")

    return correct_count, samples_count, accuracy, label_correct_count


def test_model(logger, model, dataloaders):
    timer = utils.Timer(inspect.currentframe().f_code.co_name, logger)

    correct_count, samples_count, accuracy, _ = get_model_accuracy(
        model, dataloaders["test"]
    )
    logger.info(f"[{correct_count}/{samples_count}], Accuracy: {accuracy:.2f}%")


def test_separated_model(args, logger, model, dataloaders):
    timer = utils.Timer(inspect.currentframe().f_code.co_name, logger)

    selected_neurons = load_selected_neurons(dataloaders, args.selected_neurons_file)
    closing_test(args, logger, model, dataloaders, selected_neurons)


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
        f"{dataloaders['name']} select neurons {selected_neurons}, we get the accuracy [{separate_accuracy}% - {remove_accuracy}% = {separate_accuracy - remove_accuracy:.2f}%, point {point:.2f}]"
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
                    {"params": selected_parameters, "lr": 1e-4},
                    {"params": others_parameters, "lr": 1e-5},
                ]
            )
        else:
            optimizer = optim.Adam(parameters, lr=1e-3)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=dataloaders["epoch"]
        )
    elif dataloaders["name"] == "EMNIST":
        if isinstance(parameters[0], list):
            selected_parameters = parameters[0]
            others_parameters = parameters[1]
            optimizer = optim.Adam(
                [
                    {"params": selected_parameters, "lr": 3e-4},
                    {"params": others_parameters, "lr": 3e-5},
                ]
            )
        else:
            optimizer = optim.Adam(parameters, lr=3e-3)

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
            )
        else:
            optimizer = optim.SGD(parameters, lr=5e-2, momentum=0.9)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=dataloaders["epoch"]
        )
    elif dataloaders["name"] == "TinyImageNet":
        if isinstance(parameters[0], list):
            selected_parameters = parameters[0]
            others_parameters = parameters[1]
            optimizer = optim.SGD(
                [
                    {"params": selected_parameters, "lr": 5e-3},
                    {"params": others_parameters, "lr": 5e-4},
                ],
                momentum=0.9,
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
        if i % 8 == 0 or i == dataloaders["epoch"] - 1:
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
    timer = utils.Timer(inspect.currentframe().f_code.co_name, logger)

    model = get_model(logger, dataloaders, model_path)
    logger.info("train the model")

    layers_list = model.get_layers_list(include_fc_layers=True)
    parameters = []
    for layers in layers_list:
        if (
            isinstance(layers, nn.Conv2d)
            or isinstance(layers, nn.Linear)
            or isinstance(layers, nn.BatchNorm2d)
        ):
            parameters.extend(list(layers.parameters()))
        else:
            if isinstance(layers, list):
                # continue
                parameters.extend(list(layers[0].parameters()))
            elif isinstance(layers, Sequential):
                # print(type(layers))
                # continue
                parameters.extend(list(layers.parameters()))
                """for layer in layers:
                    if isinstance(layers, nn.Conv2d) or isinstance(
                        layers, nn.BatchNorm2d
                    ):
                        parameters.extend(list(layer.parameters()))
                    else:
                        print(type(layer))"""
    # print(parameters)
    # quit()
    """parameters = []
    for layers in layers_list:
        if isinstance(layers, nn.Linear) or isinstance(layers, nn.BatchNorm2d):
            parameters.extend(list(layers.parameters()))
        else:
            parameters.extend(list(layers[0].parameters()))"""

    train_model(args, logger, model, dataloaders, parameters, model_path)


def train_and_save_percent_dataset_model(args, logger, dataloaders, percent_range):
    timer = utils.Timer(inspect.currentframe().f_code.co_name, logger)

    accuracies = []
    for percent in range(
        percent_range[0], percent_range[1] + percent_range[2], percent_range[2]
    ):
        logger.info(f"train the model with {percent}% data")

        percent_dataloaders = copy.deepcopy(dataloaders)
        data.use_partial_dataloaders(percent_dataloaders, percent=percent)

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
        quit()

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

        if index % 100 == 0:
            np_results = np.array(results)
            np.save(file_name, np_results)

    np_results = np.array(results)
    np.save(file_name, np_results)


def select_full_combination(args, logger, model, dataloaders):
    timer = utils.Timer(inspect.currentframe().f_code.co_name, logger)

    if dataloaders["name"] != "MNIST" and dataloaders["name"] != "EMNIST":
        logger.error(
            f"the dataset is {dataloaders['name']}, while this function only supports MNIST and EMNIST"
        )
        quit()

    combinations = []
    if dataloaders["name"] == "MNIST":
        for i in range(1, 3):
            for layer_1_neurons in it.combinations(range(0, 6), i):
                for j in range(i, 7):
                    for layer_2_neurons in it.combinations(range(0, 16), j):
                        selected_neurons = {1: layer_1_neurons, 2: layer_2_neurons}
                        combinations.append(selected_neurons)
    elif dataloaders["name"] == "EMNIST":
        for i in range(1, 3):
            for layer_1_neurons in it.combinations(range(0, 10), i):
                for j in range(i, 5):
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
    # since we have iterated all combinations
    # there is no need to test random selections because we can look up its accuracy directly
    # therefore, we use the imitation algorithm in /analyze_result/random_selections
    pass


def sig(x):
    return 1 / (1 + math.exp(-x))


def get_neuron_point(args, separating_accuracy, removing_accuracy):
    accuracy = args.accuracy_base

    point = -sig((accuracy - separating_accuracy) / 1) + sig(
        (separating_accuracy - removing_accuracy) / 5
    )

    return point


def select_neurons_v2(args, logger, model, dataloaders):
    timer = utils.Timer(inspect.currentframe().f_code.co_name, logger)

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
            f"[conv{layers_list[i][0].layer_index}]Separated neurons list: {selected_neurons}"
        )

        if len(selected_neurons[i + 1]) == layers_list[i][0].layer.out_channels:
            while i + 1 < len(layers_list):
                i += 1
                selected_neurons[i + 1] = range(layers_list[i][0].layer.out_channels)
            break

    closing_test(args, logger, copy.deepcopy(model), dataloaders, selected_neurons)


def select_neurons_v2_amend(args, logger, model, dataloaders, input_file, output_file):
    timer = utils.Timer(inspect.currentframe().f_code.co_name, logger)

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
            f"[conv{layers_list[i][0].layer_index}]Separated neurons list: {selected_neurons}"
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
    # the function consumes more GPU memory

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
    timer = utils.Timer(
        f"{inspect.currentframe().f_code.co_name}_v{prune_index}", logger
    )

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
            logger.fatal(f"unsupported prune algorithm index {prune_index}")
            exit()

        if args.add_factor is not None:
            select_limit = select_limit + args.add_factor
        elif args.multiply_factor is not None:
            select_limit = select_limit * args.multiply_factor

    closing_test(args, logger, copy.deepcopy(model), dataloaders, selected_neurons)


def select_neurons_v4(args, logger, model, dataloaders, prune_index):
    timer = utils.Timer(
        f"{inspect.currentframe().f_code.co_name}_v{prune_index}", logger
    )
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

    first_layer_pool = None
    if prune_index == 1:
        first_layer_pool = pruning_select_norm(
            copy.deepcopy(model), 0, first_layer_neurons_count * 2
        )
    elif prune_index == 2:
        first_layer_pool = pruning_select_fpgm(
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
            f"[conv{layers_list[i][0].layer_index}]Separated neurons list: {selected_neurons}"
        )

        if len(selected_neurons[i + 1]) == layers_list[i][0].layer.out_channels:
            while i + 1 < len(layers_list):
                i += 1
                selected_neurons[i + 1] = range(layers_list[i][0].layer.out_channels)
            break

    closing_test(args, logger, copy.deepcopy(model), dataloaders, selected_neurons)


def train_from_scratch(args, logger, dataloaders):
    timer = utils.Timer(inspect.currentframe().f_code.co_name, logger)

    dataloaders_train = copy.deepcopy(dataloaders)
    if args.recover_dataset_percent:
        data.use_partial_dataloaders(
            dataloaders_train, percent=args.recover_dataset_percent, mode="test"
        )
    elif args.recover_dataset_count:
        data.use_partial_dataloaders(
            dataloaders_train, count=args.recover_dataset_count, mode="test"
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = scheduler = None
    dataloaders_train["epoch"] = 512

    model_list = {
        "MNIST": models.SplitMNISTNet(),
        "EMNIST": models.SplitEMNISTNet(),
        "GTSRB": models.SplitGTSRBNet(),
        "CIFAR10": models.SplitCIFAR10Net(),
        "TinyImageNet": models.SplitTinyImageNet(),
    }
    model = model_list[dataloaders_train["name"]]
    model.set_layers_on_cuda()

    layers_list = model.get_layers_list(include_fc_layers=True)
    parameters = []
    for layers in layers_list:
        if isinstance(layers, nn.Linear) or isinstance(layers, nn.BatchNorm2d):
            parameters.extend(list(layers.parameters()))
        else:
            parameters.extend(list(layers[0].parameters()))

    if dataloaders_train["name"] == "MNIST":
        optimizer = optim.Adam(parameters, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=dataloaders_train["epoch"]
        )
    elif dataloaders_train["name"] == "EMNIST":
        optimizer = optim.Adam(parameters, lr=3e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=dataloaders_train["epoch"]
        )
    elif dataloaders_train["name"] == "GTSRB":
        optimizer = optim.SGD(parameters, lr=5e-2, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=dataloaders_train["epoch"]
        )
    elif dataloaders_train["name"] == "CIFAR10":
        optimizer = optim.SGD(parameters, lr=5e-2, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=dataloaders_train["epoch"]
        )
    elif dataloaders["name"] == "TinyImageNet":
        optimizer = optim.SGD(parameters, lr=5e-2, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=dataloaders_train["epoch"]
        )

    for i in tqdm(range(dataloaders_train["epoch"])):
        loss_ep = 0
        for imgs, labels in dataloaders_train["train"]:
            imgs = imgs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            scores = model(imgs)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()

            loss_ep += loss.item()
        scheduler.step()

    test_model(logger, model, dataloaders_train)


def recover_model(args, logger, model, dataloaders, model_path):
    timer = utils.Timer(inspect.currentframe().f_code.co_name, logger)

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
        dataloaders_recover,
        f"selected_neurons_{int(args.percent_factor)}%.json"
        if args.percent_factor
        else f"recover_selected_neurons.json",
    )
    model.selected_neurons = selected_neurons
    logger.info(f"selected_neurons: {selected_neurons}")
    logger.info("original accuracy")
    model.work_mode = models.WorkMode.split
    test_model(logger, model, dataloaders)

    recover_parameters = []
    others_parameters = []
    for layers in model.get_layers_list(True):
        if (
            isinstance(layers, nn.Linear)
            or isinstance(layers, nn.BatchNorm2d)
            or isinstance(layers, Sequential)
        ):
            others_parameters.extend(list(layers.parameters()))
        else:
            if isinstance(layers[0], nn.Conv2d):
                for i, layer in enumerate(layers):
                    if i in selected_neurons[2]:
                        layers[i].reset_parameters()
                        recover_parameters.extend(list(layer.parameters()))
                    else:
                        others_parameters.extend(list(layer.parameters()))
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
        [recover_parameters, [] if args.recover_freeze else others_parameters],
        recover_model_path,
    )

    logger.info("after recovering the model(ReLU)")
    test_model(logger, model, dataloaders)


def obfuscate_intermidiate_results(
    args, logger, dataloaders, output: torch.tensor, randomize: bool = True
) -> torch.tensor:
    selected_neurons = load_selected_neurons(
        dataloaders, f"selected_neurons_{args.percent_factor}%.json"
    )

    for i in range(64):
        if i in selected_neurons[2] and randomize:
            output[:, i, :, :] *= random.random() * 100
        else:
            output[:, i, :, :] = 0

    return output


def save_results(dataloaders, results: list, pictures_name: str, count: int) -> None:
    path = f"../saved_models/{dataloaders['name']}/{pictures_name}"
    output_tensor = output_tensor = (
        torch.cat((results[0][:count], results[1][:count]))
        if len(results) == 2
        else torch.cat((results[0][:count], results[1][:count], results[2][:count]))
    )

    save_image(output_tensor / 2 + 0.5, path)


def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]


def total_variation(input):
    batch_size, channel, height, width = (
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    )

    count_height = _tensor_size(input[:, :, 1:, :])
    count_width = _tensor_size(input[:, :, :, 1:])
    h_tv = torch.pow(input[:, :, 1:, :] - input[:, :, : height - 1, :], 2).sum()
    w_tv = torch.pow(input[:, :, :, 1:] - input[:, :, :, : width - 1], 2).sum()
    return (h_tv / count_height + w_tv / count_width) / batch_size


def l2loss(x):
    return (x**2).mean()


def recover_input(args, logger, model, dataloaders, pictures_name: str) -> None:
    attack_model = copy.deepcopy(model)
    attack_model.work_mode = models.WorkMode.attack_out

    input_shape = [len(dataloaders["test"].dataset)] + list(
        dataloaders["test"].dataset[0][0].shape
    )
    input = torch.empty(input_shape).fill_(0.5).requires_grad_(True)
    input_optimizer = torch.optim.Adam([input], lr=0.001, amsgrad=True)
    mse_loss = torch.nn.MSELoss()

    obfuscated_result = None
    for imgs, _ in dataloaders["test"]:
        real_result = attack_model(imgs.cuda())
        obfuscated_result = obfuscate_intermidiate_results(
            args, logger, dataloaders, real_result
        )

    # rescale
    selected_neurons = load_selected_neurons(
        dataloaders, f"selected_neurons_{args.percent_factor}%.json"
    )
    for i in range(real_result.shape[1]):
        if i in selected_neurons[2]:
            obfuscated_result[:, i, :, :] /= sum(obfuscated_result[:, i, :, :]) / sum(
                real_result[:, i, :, :]
            )

    results = []
    for i in range(1000):
        for imgs, _ in dataloaders["test"]:
            input_optimizer.zero_grad()

            guess_result = attack_model(input.cuda())
            loss = (
                mse_loss(guess_result, obfuscated_result)
                + 0.1 * total_variation(input)
                + 1 * l2loss(input)
            )
            loss.backward(retain_graph=True)

            input_optimizer.step()

            results = [imgs.detach(), input.detach()]

            if i % 100 == 0:
                save_results(
                    dataloaders,
                    results,
                    f"{args.percent_factor}_{i}_{pictures_name}",
                    8,
                )
                output_tensor = torch.cat((results[0], results[1]))
                torch.save(output_tensor, f"{args.percent_factor}.pt")


class Square(nn.Module):
    def forward(self, x):
        return torch.square(x)


class AutoEncoderCIFAR10(nn.Module):
    def __init__(self, args, dataloaders):
        super(AutoEncoderCIFAR10, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            Square(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1
            ),
            Square(),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=3, kernel_size=3, padding=1
            ),
        )

        self.selected_neurons = load_selected_neurons(
            dataloaders, f"selected_neurons_{args.percent_factor}%.json"
        )

    def forward(self, input):
        encoder_output = self.encoder(input)

        for i in range(encoder_output.shape[1]):
            if i not in self.selected_neurons[2]:
                encoder_output[:, i, :, :] = 0

        decoder_output = self.decoder(encoder_output)
        return decoder_output


def recover_input_autoencoder(
    args, logger, model, dataloaders, pictures_name: str
) -> None:
    autoencoder = AutoEncoderCIFAR10(args, dataloaders).cuda()

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters())

    for i in range(100):
        for imgs, _ in dataloaders["test"]:
            imgs = imgs.cuda()

            optimizer.zero_grad()
            output = autoencoder(imgs)
            loss = criterion(output, imgs)
            loss.backward()
            optimizer.step()

            results = [imgs.detach(), output.detach()]

            if i % 10 == 0:
                save_results(
                    dataloaders,
                    results,
                    f"autoencoder_{args.percent_factor}_{i}_{pictures_name}",
                    8,
                )

    attack_model = copy.deepcopy(model)
    attack_model.work_mode = models.WorkMode.attack_out
    real_result = None
    obfuscated_result = None
    for imgs, _ in dataloaders["test"]:
        real_result = attack_model(imgs.cuda())
        obfuscated_result = obfuscate_intermidiate_results(
            args, logger, dataloaders, real_result
        )

    # rescale
    selected_neurons = load_selected_neurons(
        dataloaders, f"selected_neurons_{args.percent_factor}%.json"
    )
    for i in range(real_result.shape[1]):
        if i in selected_neurons[2]:
            obfuscated_result[:, i, :, :] /= sum(obfuscated_result[:, i, :, :]) / sum(
                real_result[:, i, :, :]
            )

    recover_result = autoencoder.decoder(obfuscated_result)
    results.append(recover_result)

    save_results(
        dataloaders,
        results,
        f"autoencoder_{args.percent_factor}_final_{pictures_name}",
        8,
    )


def defense_weight_stealing(args, logger, model, dataloaders) -> None:
    obfuscated_retrain_model = copy.deepcopy(model)

    selected_neurons = load_selected_neurons(
        dataloaders, f"selected_neurons_{args.percent_factor}%.json"
    )
    obfuscated_layer = []
    layers_list = obfuscated_retrain_model.get_layers_list(include_fc_layers=True)
    for layer in layers_list:
        if isinstance(layer, list) and isinstance(layer[0], nn.Conv2d):
            obfuscated_layer = layer
            break

    retrain_parameters = []
    for i in range(len(obfuscated_layer)):
        if i in selected_neurons[2]:
            logger.info(f"original {i}th neuron: {obfuscated_layer[i].weight}")
            obfuscated_layer[i].reset_parameters()
            retrain_parameters.extend(list(obfuscated_layer[i].parameters()))

    test_model(logger, obfuscated_retrain_model, dataloaders)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(retrain_parameters, lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=dataloaders["epoch"]
    )

    best_loss = 1
    best_model = None

    for i in range(64):
        loss_ep = 0
        for imgs, labels in dataloaders["train"]:
            imgs = imgs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            scores = obfuscated_retrain_model(imgs)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()

            loss_ep += loss.item()

        average_loss = loss_ep / len(dataloaders["train"].dataset)
        if average_loss < best_loss:
            best_loss = average_loss
            best_model = copy.deepcopy(obfuscated_retrain_model)

        scheduler.step()
        with torch.no_grad():
            _, _, accuracy, _ = get_model_accuracy(
                obfuscated_retrain_model, dataloaders["validate"]
            )
            logger.info(
                f"[Epoch {i:3}]Loss: {average_loss:.8f}, Accuracy: {accuracy:5.3f}%"
            )

    obfuscated_retrain_model = best_model
    for i in range(len(obfuscated_layer)):
        if i in selected_neurons[2]:
            logger.info(f"retrain {i}th neuron: {obfuscated_layer[i].weight}")
    test_model(logger, obfuscated_retrain_model, dataloaders)

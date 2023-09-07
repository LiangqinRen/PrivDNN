import torch
import os
import logging
import time
import argparse
import pickle
import enum


class ModelWorkMode(enum.Enum):
    train = 1
    test = 2
    select_subset = 3
    recover = 4
    fhe_inference = 5
    something = 6


def is_file_exist(file_path):
    return os.path.exists(file_path)


def check_cuda_availability():
    if not torch.cuda.is_available():
        raise SystemExit("cuda is not available!")


def get_file_and_console_logger(args):
    LOG_FOLDER_PATH = "../log"
    log_level = args.log_level
    log_levels = {
        10: logging.DEBUG,
        20: logging.INFO,
        30: logging.WARNING,
        40: logging.ERROR,
        50: logging.FATAL,
    }

    formatter = logging.Formatter(
        fmt="[%(asctime)s.%(msecs)03d][%(filename)10s:%(lineno)4d][%(levelname)5s]|%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    current_time = time.strftime("%Y:%m:%d-%H:%M:%S", time.localtime(time.time()))
    log_path = "{}/{}.log".format(LOG_FOLDER_PATH, current_time)
    handler_to_file = logging.FileHandler(log_path, mode="w")
    handler_to_file.setFormatter(formatter)
    handler_to_console = logging.StreamHandler()
    handler_to_console.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(log_levels[log_level])
    logger.addHandler(handler_to_file)
    logger.addHandler(handler_to_console)

    return logger


def get_argparser():
    parser = argparse.ArgumentParser(description="Hello!")
    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        help="the dataset to train, test, retrain, etc.",
    )
    parser.add_argument(
        "--log_level",
        type=int,
        default=20,
        help="log level",
    )
    parser.add_argument(
        "--model_work_mode",
        type=int,
        default=None,
        help="work mode of the model",
    )
    parser.add_argument(
        "--train_dataset_percent",
        type=int,
        default=None,
        help="percent of the dataset to train the model",
    )
    parser.add_argument(
        "--selected_neurons_file",
        type=str,
        default=None,
        help="json file of the selected neurons in every layer",
    )
    parser.add_argument(
        "--selected_layers_file",
        type=str,
        default=None,
        help="json file of the selected neurons count in every layer",
    )
    parser.add_argument(
        "--initial_layer_index",
        type=int,
        default=None,
        help="initial layer neurons controls the layer to start the selection",
    )
    parser.add_argument(
        "--encrypt_layers_count",
        type=int,
        default=None,
        help="how many layers we need to select, i.e., to encrypt and protect",
    )
    parser.add_argument(
        "--initial_layer_neurons",
        type=int,
        default=None,
        help="initial layer neurons controls the neuron count in the first layer",
    )
    parser.add_argument(
        "--add_factor",
        type=int,
        default=None,
        help="add factor controls how many neurons we should select in a layer",
    )
    parser.add_argument(
        "--multiply_factor",
        type=int,
        default=None,
        help="multiply factor controls how many neurons we should select in a layer",
    )
    parser.add_argument(
        "--percent_factor",
        type=int,
        default=None,
        help="percent factor controls how many neurons we should select in a layer",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="alpha controls the weight of separating accuracy",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="beta controls the weight of accuracy difference",
    )
    parser.add_argument(
        "--accuracy_base",
        type=float,
        default=None,
        help="accuracy base controls the expected separating accuracy",
    )
    parser.add_argument(
        "--difference_base",
        type=float,
        default=None,
        help="diff base controls the expected difference between the separating accuracy and removing accuracy",
    )
    parser.add_argument(
        "--random_selection_times",
        type=int,
        default=None,
        help="random selection times controls the repeating times of select function when the algorithm is random selection",
    )
    parser.add_argument(
        "--greedy_step",
        type=int,
        default=None,
        help="greedy step controls the selected neurons count per selection",
    )
    parser.add_argument(
        "--recover_dataset_percent",
        type=int,
        default=None,
        help="percent of the trainning dataset to recover the model",
    )
    parser.add_argument(
        "--recover_dataset_count",
        type=int,
        default=None,
        help="count of the trainning dataset to recover the model",
    )

    args = parser.parse_args()
    args.model_work_mode = ModelWorkMode(args.model_work_mode)
    return args


def write_content_to_file(content, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(content, file)


def read_content_from_file(file_path):
    with open(file_path, "rb") as file:
        content = pickle.load(file)
        return content


def is_folder_empty(folder_path):
    files_list = os.listdir(folder_path)
    return len(files_list) == 0


def get_model_path(args, dataloaders, percent=100):
    train_batch_size = dataloaders["train"].batch_size
    model_path = f"../saved_models/{dataloaders['name']}/{dataloaders['name']}_{dataloaders['epoch']}_{train_batch_size}_{percent}.pth"

    return model_path
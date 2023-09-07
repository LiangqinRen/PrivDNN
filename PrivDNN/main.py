import data
import worker
import models
import utils
import time

if __name__ == "__main__":
    start_time = time.time()

    args = utils.get_argparser()
    logger = utils.get_file_and_console_logger(args)
    logger.info(
        f"Parameters from the terminal:\n\
            \tdataset: {args.dataset}\n\
            \tlog_level: {args.log_level}\n\
            \tmodel_work_mode: {args.model_work_mode}\n\
            \ttrain_dataset_percent: {args.train_dataset_percent}\n\
            \tselected_neurons_file: {args.selected_neurons_file}\n\
            \tselected_layers_file: {args.selected_layers_file}\n\
            \tinitial_layer_index: {args.initial_layer_index}\n\
            \tencrypt_layers_count: {args.encrypt_layers_count}\n\
            \tinitial_layer_neurons: {args.initial_layer_neurons}\n\
            \tadd_factor: {args.add_factor}\n\
            \tmultiply_factor: {args.multiply_factor}\n\
            \tpercent_factor: {args.percent_factor}\n\
            \talpha: {args.alpha}\n\
            \tbeta: {args.beta}\n\
            \taccuracy_base: {args.accuracy_base}\n\
            \tdifference_base: {args.difference_base}\n\
            \trandom_selection_times: {args.random_selection_times}\n\
            \tgreedy_step: {args.greedy_step}\n\
            \trecover_dataset_percent: {args.recover_dataset_percent}\n\
            \trecover_dataset_count: {args.recover_dataset_count}"
    )

    utils.check_cuda_availability()
    dataloaders_selection = {
        "MNIST": data.get_MNIST_dataloader,
        "EMNIST": data.get_EMNIST_dataloader,
        "FMNIST": data.get_FMNIST_dataloader,
        "GTSRB": data.get_GTSRB_dataloader,
        "CIFAR10": data.get_CIFAR10_dataloader,
        "CIFAR100": data.get_CIFAR100_dataloader,
    }

    logger.info("check CUDA, get argparser and logger")

    dataloaders = dataloaders_selection[args.dataset](args.model_work_mode)
    model_path = utils.get_model_path(args, dataloaders)

    if args.model_work_mode == utils.ModelWorkMode.train:
        if args.train_dataset_percent == 100:
            worker.train_and_save_model(args, logger, dataloaders, model_path)
        else:
            worker.train_and_save_percent_dataset_model(
                args, logger, dataloaders, [1, args.train_dataset_percent, 1]
            )
    elif args.model_work_mode == utils.ModelWorkMode.test:
        trained_model = worker.load_trained_model(model_path)
        if args.selected_neurons_file is None:
            layers = trained_model.get_layers_list()
            worker.test_model(logger, trained_model, dataloaders)
        else:
            worker.test_separated_model(args, logger, trained_model, dataloaders)
    elif args.model_work_mode == utils.ModelWorkMode.select_subset:
        trained_model = worker.load_trained_model(model_path)
        # worker.select_neurons_v1(args, logger, trained_model, dataloaders)
        # worker.select_neurons_v1_multi(args, logger, trained_model, dataloaders)
        worker.select_neurons_v2(args, logger, trained_model, dataloaders)
        logger.info(
           f"select_neurons_v2 running time: {time.time() - start_time:.3f} seconds"
        )
        start_time = time.time()
        worker.select_neurons_v3(args, logger, trained_model, dataloaders, 1)
        logger.info(
            f"select_neurons_v3_1 running time: {time.time() - start_time:.3f} seconds"
        )
        start_time = time.time()
        worker.select_neurons_v3(args, logger, trained_model, dataloaders, 2)
        logger.info(
            f"select_neurons_v3_2 running time: {time.time() - start_time:.3f} seconds"
        )
        start_time = time.time()
        worker.select_neurons_v3(args, logger, trained_model, dataloaders, 3)
        logger.info(
            f"select_neurons_v3_3 running time: {time.time() - start_time:.3f} seconds"
        )
        start_time = time.time()
        worker.select_neurons_v3(args, logger, trained_model, dataloaders, 4)
        logger.info(
            f"select_neurons_v3_4 running time: {time.time() - start_time:.3f} seconds"
        )
        start_time = time.time()
        worker.select_neurons_v4(args, logger, trained_model, dataloaders, 1)
        logger.info(
            f"select_neurons_v4_1 running time: {time.time() - start_time:.3f} seconds"
        )
        start_time = time.time()
        worker.select_neurons_v4(args, logger, trained_model, dataloaders, 2)
        logger.info(
            f"select_neurons_v4_2 running time: {time.time() - start_time:.3f} seconds"
        )
        # start_time = time.time()
        # worker.select_full_combination(args, logger, trained_model, dataloaders)
    elif args.model_work_mode == utils.ModelWorkMode.recover:
        trained_model = worker.load_trained_model(model_path)
        worker.recover_model(args, logger, trained_model, dataloaders, model_path)
    elif args.model_work_mode == utils.ModelWorkMode.fhe_inference:
        trained_model = worker.load_trained_model(model_path)
        trained_model.work_mode = models.WorkMode.cipher

        logger.info("SEAL separate inference:")
        worker.test_model(logger, trained_model, dataloaders)
        logger.info(
            f"SEAL separate inference running time: {time.time() - start_time:.3f} seconds"
        )

        start_time = time.time()
        trained_model.cpp_work_mode = models.CppWorkMode.remove
        logger.info("SEAL remove inference:")
        worker.test_model(logger, trained_model, dataloaders)
        logger.info(
            f"SEAL remove inference running time: {time.time() - start_time:.3f} seconds"
        )
    elif args.model_work_mode == utils.ModelWorkMode.something:
        trained_model = worker.load_trained_model(model_path)
    else:
        raise Exception("Unknown model_work_mode")

    end_time = time.time()
    logger.info(f"Running time: {end_time - start_time:.3f} seconds")
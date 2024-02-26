import data
import worker
import models
import utils
import time

if __name__ == "__main__":
    start_time = time.time()

    utils.check_cuda_availability()

    args = utils.get_argparser()
    logger = utils.get_file_and_console_logger(args)
    utils.show_parameters(args, logger)

    dataloaders_selection = {
        "MNIST": data.get_MNIST_dataloader,
        "EMNIST": data.get_EMNIST_dataloader,
        "GTSRB": data.get_GTSRB_dataloader,
        "CIFAR10": data.get_CIFAR10_dataloader,
        "TINYIMAGENET": data.get_TinyImageNet_dataloader,
    }

    dataloaders = dataloaders_selection[args.dataset]()
    model_path = utils.get_model_path(args, dataloaders)
    if args.work_mode == utils.WorkMode.train:
        worker.train_and_save_model(args, logger, dataloaders, model_path)
    elif args.work_mode == utils.WorkMode.test:
        trained_model = worker.load_trained_model(model_path)
        if args.sub_work_mode == 0:
            worker.test_model(args, logger, trained_model, dataloaders)
        elif args.sub_work_mode == 1:
            worker.test_separated_model(args, logger, trained_model, dataloaders)
        else:
            raise Exception("Unknown sub_work_mode")
    elif args.work_mode == utils.WorkMode.select_subset:
        trained_model = worker.load_trained_model(model_path)

        # worker.select_neurons_v1(args, logger, trained_model, dataloaders)
        worker.select_neurons_v2(args, logger, trained_model, dataloaders)
        """worker.select_neurons_v2_amend(
            args,
            logger,
            trained_model,
            dataloaders,
            f"selected_neurons_{args.percent_factor - 5}%.json",
            f"selected_neurons_{args.percent_factor}%.json",
        )"""

        # worker.select_neurons_v3(args, logger, trained_model, dataloaders, 1)
        # worker.select_neurons_v3(args, logger, trained_model, dataloaders, 2)
        # worker.select_neurons_v3(args, logger, trained_model, dataloaders, 3)
        # worker.select_neurons_v3(args, logger, trained_model, dataloaders, 4)

        # worker.select_neurons_v4(args, logger, trained_model, dataloaders, 1)
        # worker.select_neurons_v4(args, logger, trained_model, dataloaders, 2)

        # worker.select_full_combination(args, logger, trained_model, dataloaders)
    elif args.work_mode == utils.WorkMode.recover:
        trained_model = worker.load_trained_model(model_path)
        worker.recover_model(args, logger, trained_model, dataloaders, model_path)
        # worker.train_from_scratch(args, logger, dataloaders)
        # worker.recover_input(args, logger, trained_model, dataloaders, "attack.png")
        # worker.recover_input_autoencoder(
        #    args, logger, trained_model, dataloaders, "attack.png"
        # )
        # worker.defense_weight_stealing(args, logger, trained_model, dataloaders)
    elif args.work_mode == utils.WorkMode.fhe_inference:
        trained_model = worker.load_trained_model(model_path)
        trained_model.work_mode = models.WorkMode.cipher

        trained_model.cpp_work_mode = models.CppWorkMode.separate
        logger.info("SEAL separate inference:")
        worker.test_model(args, logger, trained_model, dataloaders)

        """trained_model.cpp_work_mode = models.CppWorkMode.remove
        logger.info("SEAL remove inference:")
        worker.test_model(args, logger, trained_model, dataloaders)"""

        """if args.dataset == "MNIST":
            model_path = model_path.replace(".pth", "_cpp.pth")
        trained_model = worker.load_trained_model(model_path)
        trained_model.work_mode = models.WorkMode.cipher
        trained_model.cpp_work_mode = models.CppWorkMode.full
        logger.info("SEAL full inference:")
        worker.test_model(args, logger, trained_model, dataloaders)"""
    elif args.work_mode == utils.WorkMode.something:
        trained_model = worker.load_trained_model(model_path)
    else:
        raise Exception("Unknown work_mode")

    logger.info(f"PrivDNN costs {time.time() - start_time:.3f} seconds")

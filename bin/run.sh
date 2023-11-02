#!/bin/bash

dataset="CIFAR10"

declare -A accuracy
accuracy=(["MNIST"]='99.36' ["EMNIST"]='92.62' ["GTSRB"]='93.24' ["CIFAR10"]='89.12')


if [[ $1 == 'train' ]]
then
    # train the model from scratch with 100% train set
    python ../PrivDNN/main.py --dataset $dataset --model_work_mode 1 --train_dataset_percent 100
elif [[ $1 == 'test' ]]
then
    # test the normal model performance
    python ../PrivDNN/main.py --dataset $dataset --model_work_mode 2

    # test the model performance with selected filters
    # python ../PrivDNN/main.py --dataset $dataset --model_work_mode 2 --selected_neurons_file "selected_neurons_100%.json" --accuracy_base ${accuracy[$dataset]}
elif [[ $1 == 'select' ]]
then
    # select filters, the main file decides the algorithm to use
    python ../PrivDNN/main.py --dataset $dataset --model_work_mode 3 --initial_layer_index 0 --encrypt_layers_count 2 --initial_layer_neurons 1 --add_factor 0 --accuracy_base ${accuracy[$dataset]} --greedy_step 1

    # python ../PrivDNN/main.py --dataset $dataset --model_work_mode 3 --initial_layer_index 0 --encrypt_layers_count 2 --percent_factor 5 --accuracy_base ${accuracy[$dataset]} --greedy_step 1
elif [[ $1 == 'recover' ]]
then
    # recover the model with different amount of pictures
    for count in {100,250,500,1000}
    do
        python ../PrivDNN/main.py --dataset $dataset --model_work_mode 4 --recover_dataset_count $count
    done
elif [[ $1 == 'inference' ]]
then
    # inference the model in the cipher domain with the SEAL library
    python ../PrivDNN/main.py --dataset $dataset --model_work_mode 5
elif [[ $1 == 'clean' ]]
then
    # clean cipher domain data files
    find ../seal/data -type f | xargs rm -rf
elif [[ $1 == '' ]]
then
    # debug
    echo "Have a nice day!"
fi
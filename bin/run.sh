#!/bin/bash

dataset=$1
function=$2

# convert lowercase dataset to uppercase
dataset=`echo $dataset | tr '[:lower:]' '[:upper:]'`

declare -A accuracy
accuracy=(["MNIST"]='99.36' ["EMNIST"]='93.08' ["GTSRB"]='93.51' ["CIFAR10"]='90.76' ["TINYIMAGENET"]='72.00')


if [[ $function == 'train' ]]
then
    # train the model from scratch with 100% train set
    python ../PrivDNN/main.py --dataset $dataset --model_work_mode 1 --train_dataset_percent 100 --top_k_accuracy 5
elif [[ $function == 'test' ]]
then
    # test the normal model performance
    python ../PrivDNN/main.py --dataset $dataset --model_work_mode 2 --top_k_accuracy 5

    # test the model performance with selected filters
    # python ../PrivDNN/main.py --dataset $dataset --model_work_mode 2 --selected_neurons_file "selected_neurons_100%.json" --accuracy_base ${accuracy[$dataset]} 
elif [[ $function == 'select' ]]
then
    python ../PrivDNN/main.py --dataset $dataset --model_work_mode 3 --initial_layer_index 0 --encrypt_layers_count 2 --accuracy_base ${accuracy[$dataset]} --greedy_step 1 --top_k_accuracy 5 --percent_factor 15

    python ../PrivDNN/main.py --dataset $dataset --model_work_mode 3 --initial_layer_index 0 --encrypt_layers_count 2 --accuracy_base ${accuracy[$dataset]} --greedy_step 1 --top_k_accuracy 5 --percent_factor 25

    exit
    # select filters, the main file decides the algorithm to use
    # python ../PrivDNN/main.py --dataset $dataset --model_work_mode 3 --initial_layer_index 0 --encrypt_layers_count 2 --initial_layer_neurons 1 --add_factor 0 --accuracy_base ${accuracy[$dataset]} --greedy_step 1
    for count in {10..100..5}
    do 
        python ../PrivDNN/main.py --dataset $dataset --model_work_mode 3 --initial_layer_index 0 --encrypt_layers_count 2 --percent_factor $count --accuracy_base ${accuracy[$dataset]} --greedy_step 1 --top_k_accuracy 5
    done
    # python ../PrivDNN/main.py --dataset $dataset --model_work_mode 3 --initial_layer_index 0 --encrypt_layers_count 2 --percent_factor 50 --accuracy_base ${accuracy[$dataset]} --greedy_step 1
elif [[ $function == 'recover' ]]
then
    python ../PrivDNN/main.py --dataset $dataset --model_work_mode 4 --recover_dataset_count 1000  --top_k_accuracy 5
    exit
    for count in {50..100..5}
    do
        python ../PrivDNN/main.py --dataset $dataset --model_work_mode 4 --recover_dataset_count 1000 --percent_factor $count --top_k_accuracy 5
        # python ../PrivDNN/main.py --dataset $dataset --model_work_mode 4 --recover_dataset_count 1000 --percent_factor $count --recover_freeze --top_k_accuracy 5
    done
    exit

    # recover the model with different amount of pictures
    for count in {100,250,500,1000}
    do
        python ../PrivDNN/main.py --dataset $dataset --model_work_mode 4 --recover_dataset_count $count
    done
elif [[ $function == 'inference' ]]
then
    # inference the model in the cipher domain with the SEAL library
    python ../PrivDNN/main.py --dataset $dataset --model_work_mode 5 --top_k_accuracy 5
    # python ../PrivDNN/main.py --dataset $dataset --model_work_mode 5 --percent 10
elif [[ $function == 'clean' ]]
then
    # clean cipher domain data files
    find ../seal/data -type f | xargs rm -rf
elif [[ $function == '' ]]
then
    # debug
    echo "Have a nice day!"
fi
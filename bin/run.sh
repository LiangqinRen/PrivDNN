#!/bin/bash

dataset=$1
workmode=$2
submode=$3

# convert lowercase dataset to uppercase
dataset=`echo $dataset | tr '[:lower:]' '[:upper:]'`

declare -A accuracy
accuracy=(["MNIST"]='99.36' ["EMNIST"]='93.08' ["GTSRB"]='93.51' ["CIFAR10"]='90.76' ["TINYIMAGENET"]='72.00')

declare -A top_k_accuracy
top_k_accuracy=(["MNIST"]='1' ["EMNIST"]='1' ["GTSRB"]='1' ["CIFAR10"]='1' ["TINYIMAGENET"]='5')

if [[ $workmode == 'train' ]]
then
    # train the model from scratch with 100% train set
    python ../PrivDNN/main.py --dataset $dataset --work_mode 1 --train_dataset_percent 100 --top_k_accuracy ${top_k_accuracy[$dataset]}
elif [[ $workmode == 'test' ]]
then
    if [[ $submode == '' ||  $submode == '0' ]]
    then
        # test the model accuracy
        python ../PrivDNN/main.py --dataset $dataset --work_mode 2 --sub_work_mode $submode --top_k_accuracy ${top_k_accuracy[$dataset]}
    elif [[ $submode == '1' ]]
    then
        # test the model accuracy with selected filters
        python ../PrivDNN/main.py --dataset $dataset --work_mode 2 --sub_work_mode $submode --top_k_accuracy ${top_k_accuracy[$dataset]} --selected_neurons_file "selected_neurons.json" --accuracy_base ${accuracy[$dataset]} 
    fi
elif [[ $workmode == 'select' ]]
then
    if [[ $dataset == 'MNIST' ]]
    then
        python ../PrivDNN/main.py --dataset $dataset --work_mode 3 --initial_layer_index 0 --encrypt_layers_count 2 --initial_layer_neurons 1 --add_factor 0 --accuracy_base ${accuracy[$dataset]} --greedy_step 1
    elif [[ $dataset == 'EMNIST' ]]
    then
        python ../PrivDNN/main.py --dataset $dataset --work_mode 3 --initial_layer_index 0 --encrypt_layers_count 2 --initial_layer_neurons 1 --add_factor 0 --accuracy_base ${accuracy[$dataset]} --greedy_step 1
    else
        pass
    fi

    # select filters, the main file decides the algorithm to use
    

    # python ../PrivDNN/main.py --dataset $dataset --work_mode 3 --initial_layer_index 0 --encrypt_layers_count 2 --percent_factor 50 --accuracy_base ${accuracy[$dataset]} --greedy_step 1
elif [[ $workmode == 'recover' ]]
then
    # recover models, the main file decides the action to execute
    python ../PrivDNN/main.py --dataset $dataset --work_mode 4 --recover_dataset_count 1000  --top_k_accuracy ${top_k_accuracy[$dataset]}
elif [[ $workmode == 'inference' ]]
then
    # inference the model in the cipher domain with the SEAL library
    python ../PrivDNN/main.py --dataset $dataset --work_mode 5 --top_k_accuracy ${top_k_accuracy[$dataset]}
elif [[ [$dataset == 'clean'] || [$workmode == 'clean'] ]]
then
    # clean cipher domain data files
    find ../seal/data -type f | xargs rm -rf
elif [[ $workmode == '' ]]
then
    # debug
    echo "Have a nice day!"
fi
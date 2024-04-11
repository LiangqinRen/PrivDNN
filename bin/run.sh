#!/bin/bash

dataset=$1
workmode=$2
submode=$3

# convert lowercase dataset to uppercase
dataset=`echo $dataset | tr '[:lower:]' '[:upper:]'`
# convert uppercase workmode to lowercase
workmode=`echo $workmode | tr '[:upper:]' '[:lower:]'`

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
    if [[ $submode == '0' ]]
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
    if [[ $dataset == 'MNIST' || $dataset == 'EMNIST' ]]
    then
        # select filters of exact numbers
        python ../PrivDNN/main.py --dataset $dataset --work_mode 3 --sub_work_mode $submode --top_k_accuracy ${top_k_accuracy[$dataset]} --initial_layer_index 0 --encrypt_layers_count 2 --initial_layer_neurons 1 --add_factor 0 --accuracy_base ${accuracy[$dataset]} --greedy_step 1
    else
        # select filters of a percentage
        python ../PrivDNN/main.py --dataset $dataset --work_mode 3 --sub_work_mode $submode --top_k_accuracy ${top_k_accuracy[$dataset]} --initial_layer_index 0 --encrypt_layers_count 2 --percent_factor 5 --accuracy_base ${accuracy[$dataset]} --greedy_step 1
    fi
elif [[ $workmode == 'recover' ]]
then
    if [[ $submode == '0' ]]
    then
        # train the model from scratch, recover_dataset_count decides the training sample count
        python ../PrivDNN/main.py --dataset $dataset --work_mode 4 --sub_work_mode $submode --top_k_accuracy ${top_k_accuracy[$dataset]} --recover_dataset_count 1000
    elif [[ $submode == '1' ]]
    then
        # recover model
        python ../PrivDNN/main.py --dataset $dataset --work_mode 4 --sub_work_mode $submode --top_k_accuracy ${top_k_accuracy[$dataset]} --recover_dataset_count 1000 --accuracy_base ${accuracy[$dataset]}
    elif [[ $submode == '2' ]]
    then
        # recover input of CIFAR10
        # percent_factor decides the selected neurons json file, i.e, selected_neurons_{percent_factor}%.json
        python ../PrivDNN/main.py --dataset $dataset --work_mode 4 --sub_work_mode $submode --top_k_accuracy ${top_k_accuracy[$dataset]} --percent_factor 50
    elif [[ $submode == '3' ]]
    then
        # polymorphic obfuscation of CIFAR10
        python ../PrivDNN/main.py --dataset $dataset --work_mode 4 --sub_work_mode $submode --top_k_accuracy ${top_k_accuracy[$dataset]} --percent_factor 50
    fi
elif [[ $workmode == 'inference' ]]
then
    # inference the model in the cipher domain with the SEAL library
    python ../PrivDNN/main.py --dataset $dataset --work_mode 5 --top_k_accuracy ${top_k_accuracy[$dataset]}
elif [[ [$dataset == 'clean'] || [$workmode == 'clean'] ]]
then
    # clean cipher domain data files
    find ../seal/data -type f | xargs rm -rf
else
    # debug
    echo "Unrecognized parameters, have a nice day!"
fi
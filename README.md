### PrivDNN: A Secure Multi-Party Computation Framework for Deep Learning using Partial DNN Encryption

---

<img src="pictures/privdnn.png" alt="drawing" width="500"/>

---

### Quick Start
```
# Our key idea is to select and encrypt partial crucial filters to speed up the FHE inference
# PrivDNN's select function supports four algorithms
bash bin/run.sh select
```
---

## Table of Contents

[toc]

### Requirements

- Python 3.10.4, PyTorch 2.1.0, CUDA 11.7
- Find detailed packages at *requirements.txt* 

### Well Trained Models

PrivDNN uses four datasets, i.e., MNIST, EMNIST, GTSRB, and CIFAR-10. We train all models from scratch and execute experiments based on the well-trained models. The models we use can be found at 
[Google Drive](https://drive.google.com/drive/folders/15vXR91hg6reWBr-DBrMjz__W5c54w8Nm?usp=sharing).

### Functions Explanation

We can use the script **run.sh** to run any function, such as 

```
bash bin/run.sh train
```

#### train

```
python ../PrivDNN/main.py --dataset $dataset --model_work_mode 1 --train_dataset_percent 100
```

Train the model of the dataset. If the program finds the existing model, continue the training; otherwise, train from scratch. **train_dataset_percent** determines the percentage of the trainset to train the model.

#### test

```
python ../PrivDNN/main.py --dataset $dataset --model_work_mode 2
```

Test the model accuracy performance.

```
python ../PrivDNN/main.py --dataset $dataset --model_work_mode 2 --selected_neurons_file "recover_selected_neurons.json" --accuracy_base ${accuracy[$dataset]}
```

Test the model accuracy performance with selected filters. **selected_neurons_file** determines the selected filter file, a JSON file indicating every layer's selected filters.

#### select

```
python ../PrivDNN/main.py --dataset $dataset --model_work_mode 3 --initial_layer_index 0 --encrypt_layers_count 2 --initial_layer_neurons 1 --add_factor 0 --accuracy_base ${accuracy[$dataset]} --greedy_step 1
```

Select filters with four algorithms. **initial_layer_index** determines the first convolutional layer to select filters. All experiments start with 0. **encrypt_layers_count** determines the layers to select, and all experiments select 2 layers. **initial_layer_neurons** and **add_factor** determine how many filters to select, i.e., the first layer selects **initial_layer_neurons** filters, and the second layer selects **initial_layer_neurons + add_factor** filters. We can also use the **multiply_factor** to select **initial_layer_neurons \* multiply_factor** filters at the second layer. **accuracy_base** determines the original accuracy of models, which calculates the selection point. **greedy_step** determines how many filters to select after the point sorting in the greedy algorithm.

```
python ../PrivDNN/main.py --dataset $dataset --model_work_mode 3 --initial_layer_index 0 --encrypt_layers_count 2 --percent_factor 10 --accuracy_base ${accuracy[$dataset]} --greedy_step 1
```

We can also use the **percent_factor** to indicate that the first and second layers select the same percentage of filters.

#### recover

```
python ../PrivDNN/main.py --dataset $dataset --model_work_mode 4 --recover_dataset_count $count
```

Recover the model with selected filters. **recover_dataset_count** determines the picture count from the test to recover the model. We can also use **recover_dataset_percent** to determine the percentage of trainsets to retrain the model. The default selected filters file is "recover_selected_neurons.json" in "saved_models/dataset/".

#### inference

```
python ../PrivDNN/main.py --dataset $dataset --model_work_mode 5
```

Call the C++ SEAL cipher domain inference. The paper only supports the MNIST dataset. The default selected filters file is "inference_encrypted_neurons.json" in "saved_models/dataset/".

#### clean

```
bash run.sh clean
```

This function cleans the encrypted model data in the SEAL inference parts. The C++ part will check if the file exists. If it exists, it will pass the generating process and use it. If we change the SEAL parameters, we must run this command to clean encrypted model data, and the program will generate new data automatically.

### Citation
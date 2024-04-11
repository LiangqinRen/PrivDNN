<!-- omit in toc -->
### PrivDNN: A Secure Multi-Party Computation Framework for Deep Learning using Partial DNN Encryption

---
**Abstract**: In the past decade, we have witnessed an exponential growth of deep learning algorithms, models, platforms, and applications. While existing DL applications and Machine learning as a service (MLaaS) frameworks assume a fully trusted model, the need for privacy-preserving DNN evaluation arises. In a secure multi-party computation scenario, both the model and the dataset are considered proprietary, i.e., the model owner does not want to reveal the highly valuable DL model to the user, while the user does not wish to disclose their private data samples either. Conventional privacy-preserving deep learning solutions ask the users to send encrypted samples to the model owners, who must handle the heavy lifting of ciphertext-domain computation with homomorphic encryption. In this paper, we present a novel solution, namely, PrivDNN, which (1) offloads the computation to the user side by sharing an encrypted deep learning model with them, (2) significantly improves the efficiency of DNN evaluation using partial DNN encryption, (3) ensures model accuracy and model privacy using a core neuron selection and encryption scheme. Experimental results show that PrivDNN reduces privacy-preserving DNN inference time and memory requirement by up to 97% while maintaining model performance and security.

![PrivDNN](pictures/privdnn.png)

---

<!-- omit in toc -->
### Contents

- [Description](#description)
- [Getting Started](#getting-started)
  - [pre-trained models](#pre-trained-models)
  - [PyTorch environment](#pytorch-environment)
  - [SEAL environment](#seal-environment)
- [Usage](#usage)
  - [train](#train)
  - [test](#test)
  - [select](#select)
  - [recover](#recover)
  - [inference](#inference)
  - [clean](#clean)
  - [others](#others)
- [Citation](#citation)

### Description

This repository is the official repository of PrivDNN, a framework to accelerate Fully Homomorphic Encryption (FHE) DNN inference by reducing required cipher operations. The repository includes the necessary codes for generating the data in all tables or diagrams in the paper. We fixed the data partition seed as 0 in the dataloader and offered pre-trained models to help reproduce our findings in the paper. However, some experiments, such as running time, will have valid fluctuations, and the exact running time depends on the hardware's performance.

### Getting Started

We implemented PrivDNN using Python 3.10.13, PyTorch 2.1.0, CUDA 12.1, C++ 11.4.0, and CMake 3.26.4. We execute all experiments on a desktop computer with Ubuntu 22.04 LTS running on AMD Ryzen 7 3700X 8-core CPU, NVIDIA 3090 GPU, and 64 GB memory. To support DNN evaluation in the ciphertext domain, we adopted [Microsoft SEAL](https://github.com/microsoft/SEAL.git) 4.1.1 library.

#### pre-trained models

PrivDNN uses five datasets and corresponding models, i.e., MNIST (LeNet5), EMNIST (LeNet5), GTSRB (AlexNet), CIFAR-10 (VGG16), and Tiny ImageNet (ResNet18). We trained all models from scratch and executed experiments based on the well-trained models. The models can be found at [Google Drive](https://drive.google.com/drive/folders/15vXR91hg6reWBr-DBrMjz__W5c54w8Nm?usp=sharing).

The model should be placed into the corresponding folder as follows:

| Models Name                                     | Folder                              |
|-------------------------------------------------|-----------------------------------|
| MNIST_128_128_100.pth| PrivDNN/saved_models/MNIST        |
 MNIST_128_128_100_cpp.pth | PrivDNN/saved_models/MNIST        |
| EMNIST_128_128_100.pth                          | PrivDNN/saved_models/EMNIST       |
| GTSRB_128_128_100.pth                           | PrivDNN/saved_models/GTSRB        |
| CIFAR10_128_128_100.pth                         | PrivDNN/saved_models/CIFAR10      |
| TinyImageNet_128_128_100.pth                    | PrivDNN/saved_models/TinyImageNet |
| MNIST.npy                    | PrivDNN/analyze_result/full_combinations |
| EMNIST.npy                    | PrivDNN/analyze_result/full_combinations |
| EMNIST.tar                   | PrivDNN/data |
| tiny-imagenet-200.tar                   | PrivDNN/data |

After putting all files to the responding folders, we also need to unpack the data file in the data folder with the following commands:

>tar -xf EMNIST.tar  
tar -xf tiny-imagenet-200.tar

#### PyTorch environment

We highly recommend readers use Conda and pip to manage the PyTorch environments with the following commands:

>conda create --name privdnn python=3.10 && conda activate privdnn  
pip install -r requirements.txt

#### SEAL environment

The SEAL environment configuration is only used for the cipher inference. If readers want to test other functions first, such as selecting critical neurons, readers can skip this step.

The cipher inference may require a significant amount of computing resources and may need a long running time depending on the number of cipher neurons and hardware's performance. Please refer to [Microsoft SEAL](https://github.com/microsoft/SEAL/tree/main) for the SEAL environment configuration.

### Usage

PrivDNN offers a script *PrivDNN/bin/run.sh* to facilitate usage. The script mainly includes six functions: train, test, select, recover, inference, and clean. PrivDNN will record all experiment logs in the *log* folder named with the running time. The script accepts the dataset and function (sub-function) as parameters. Readers must use the exact dataset and function names as follows (case-insensitive):

>[mnist, emnist, gtsrb, cifar10, tinyimagenet]  
[train, test, select, inference, recover, clean]

We briefly introduce those functions as follows. For every function, there are detailed parameter explanations in the script.

#### train

The *train* function trains the model from scratch. We have offered the pre-trained models used for our experiments so readers can use our models to execute the experiments. If readers want to train the model, such as MNIST (LeNet5), from scratch, they can use the following commands. However, the reader should back up the pre-trained model before training because PrivDNN will automatically continue the training with the default model file name, like *MNIST_128_128_100.pth*.

```shell
bash run.sh mnist train
```

#### test

The *test* function tests the model's accuracy. During the test, we used the top-5 accuracy for Tiny ImageNet and the top-1 accuracy for others.

The *test* function has two modes, and the reader should select one from them. Mode 0 tests the model's original accuracy, i.e., $A_o$ in the paper. In the paper, mode 1 tests the model accuracy when selecting neurons, i.e., $A_s$ and $A_r$. PrivDNN get the selected neurons from the file *PrivDNN/saved_models/[dataset]/selected_neurons.json*

```shell
bash run.sh mnist test 0/1
```

The *test* function's results are used in Table 1.

#### select

The *select* function is the most important in PrivDNN. PrivDNN offers four algorithms to select critical neurons: random selection, greedy selection, pruning selection, and pruning+greedy selection. For datasets MNIST and EMNIST, we select an exact number of neurons in the first two layers; for GTSRB, CIFAR10, and Tiny ImageNet, we select a percentage of neurons in the first two layers. We also explain the parameters in the script.

The *select* function has five modes. The first four are corresponded to 4 algorithms. PrivDNN will save the selected neurons to the file *PrivDNN/saved_models/[dataset]/selected_neurons.json*. Mode 4 will test all possible selections as the ground truth for MNIST and EMNIST, which takes about three days in our environment. Every algorithm has some specific approaches, such as PFEC and FPGM, and we execute only one approach during the experiments to avoid the affection of cache, etc. We don't implement random selection here because we have the ground truth for all possible selections of MNIST and EMNIST. We will explain this later.

```shell
bash run.sh mnist select 0/1/2/3/4
```

The *select* function's results are used in Tables 2, 3, 4, 5 and Figures 4, 7.

#### recover

The *recover* function recovers the model with selected neurons or trains it from scratch. It also includes experiments on recovering the input and generating polymorphic obfuscation.

The *select* function has four modes. Mode 0 is training the model from scratch, i.e., $A_t$. Mode 1 is recovering the model, i.e., $A_{rec}$. Mode 2 is recovering the input. Mode 3 is generating polymorphic obfuscation.

The *recover* function's results are used in Table 7 and Figures 7, 8, and 9.

```shell
bash run.sh mnist recover 0/1/2/3
```

#### inference

The *inference* function uses the C++ SEAL library to execute the cipher domain inference. PrivDNN will use the file *PrivDNN/saved_models/[dataset]/inference_encrypted_neurons.json* as the selected neurons. The running time of the *inference* function highly depends on the dataset and selected neuron count. During this process, it may take up all CPU cores, generate large cipher files, and consume lots of memory (virtual memory). To test the function, readers can infer the MNIST dataset with selected neurons of {"1": [0], "2": [0]}, which should finish in five minutes.

To execute the inference experiments, readers must compile the C++ codes to generate related libs.

```shell
cd seal/src/interact  
mkdir build && cd build  
cmake .. && make -j
```

The *inference* function's results are used in Figures 5 and 7.

```shell
bash run.sh mnist inference
```

#### clean

```shell
bash run.sh clean
```

The *clean* function deletes the generated cipher parameters. PrivDNN will check if the file exists. It will pass the generating process and use the existing parameters if it exists. Otherwise, it will generate the required cipher parameters automatically. If readers change the SEAL parameters, they must compile and clean encrypted model data to create new data.

#### others

We have some Python programs in *PrivDNN/analyze_result* to statistic the results from previous experiments, such as the random selection algorithm. The results of those programs are used in Tables 2, 3, and Figures 5, 7.

### Citation

Liangqin Ren, Zeyan Liu, Fengjun Li, Kaitai Liang, Zhu Li, and Bo Luo. PrivDNN: A Secure Multi-Party Computation Framework for Deep Learning using Partial DNN Encryption. In the 24th Privacy Enhancing Technologies Symposium (PETS), Bristol, UK, 2024.

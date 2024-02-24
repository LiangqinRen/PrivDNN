### PrivDNN: A Secure Multi-Party Computation Framework for Deep Learning using Partial DNN Encryption

---
Abstract: In the past decade, we have witnessed an exponential growth of deep learning algorithms, models, platforms, and applications. While existing DL applications and Machine learning as a service (MLaaS) frameworks assume a fully trusted model, the need for privacy-preserving DNN evaluation arises. In a secure multi-party computation scenario, both the model and the dataset are considered proprietary, i.e., the model owner does not want to reveal the highly valuable DL model to the user, while the user does not wish to disclose their private data samples either. Conventional privacy-preserving deep learning solutions ask the users to send encrypted samples to the model owners, who must handle the heavy lifting of ciphertext-domain computation with homomorphic encryption. In this paper, we present a novel solution, namely, PrivDNN, which (1) offloads the computation to the user side by sharing an encrypted deep learning model with them, (2) significantly improves the efficiency of DNN evaluation using partial DNN encryption, (3) ensures model accuracy and model privacy using a core neuron selection and encryption scheme. Experimental results show that PrivDNN reduces privacy-preserving DNN inference time and memory requirement by up to 97% while maintaining model performance and security.


<img src="pictures/privdnn.png" alt="drawing" width="500"/>

---

### Contents
- [Getting Started](#getting-started)
  - [Pretrained Models](#pretrained-models)
  - [PyTorch Environment](#pytorch-environment)
  - [SEAL Environment](#seal-environment)
- [Usage](#usage)
  - [train](#train)
  - [test](#test)
  - [select](#select)
  - [recover](#recover)
  - [inference](#inference)
  - [clean](#clean)
  - [others](#others)
- [Citation](#citation)

### Getting Started

We implement PrivDNN using Python 3.10.13, PyTorch 2.1.0, and CUDA 12.1. All the experiments are performed on a desktop computer with Ubuntu 22.04 LTS running on AMD Ryzen 7 3700X eight-core CPU, NVIDIA 3090 GPU, and 64 GB memory. To support DNN evaluation in the ciphertext domain, we adopt [Microsoft SEAL](https://github.com/microsoft/SEAL.git) 4.1.1 library.

#### Pretrained Models

PrivDNN uses five datasets and corresponding models, i.e., MNIST (LeNet5), EMNIST (LeNet5), GTSRB (AlexNet), CIFAR-10 (VGG16) and Tiny ImageNet (ResNet18). We train all models from scratch and execute experiments based on the well-trained models. The models we use can be found at [Google Drive](https://drive.google.com/drive/folders/15vXR91hg6reWBr-DBrMjz__W5c54w8Nm?usp=sharing).

The model should be put into the corresponding folder as following.

| Models Name                                     | Folder                              |
|-------------------------------------------------|-----------------------------------|
| MNIST_128_128_100.pth| PrivDNN/saved_models/MNIST        |
 MNIST_128_128_100_cpp.pth | PrivDNN/saved_models/MNIST        |
| EMNIST_128_128_100.pth                          | PrivDNN/saved_models/EMNIST       |
| GTSRB_128_128_100.pth                           | PrivDNN/saved_models/GTSRB        |
| CIFAR10_128_128_100.pth                         | PrivDNN/saved_models/CIFAR10      |
| TinyImageNet_128_128_100.pth                    | PrivDNN/saved_models/TinyImageNet |

#### PyTorch Environment

We highly recommend you to use Conda and pip to manage the PyTorch environments with the following commands:

>conda create --name privdnn python=3.10 && conda activate privdnn  
pip install -r requirements.txt

#### SEAL Environment

The SEAL environment configuration is only used for the cipher inference. If you want to test other functions first, such as selecting critical neurons, you can pass this step.

Please refer to [Microsft SEAL](https://github.com/microsoft/SEAL/tree/main) for the SEAL environment configuration.

### Usage

PrivDNN offers a script **PrivDNN/bin/run.sh** to facilitate the usage. The script mainly includes six functions: train, test, select, recover, inference and clean.

We briefly introduce those functions as followings. For every function, there are detailed parameters explanation in the script.

#### train

*train* function is used to train the model from scratch. We have offered the pre-trained models used for our experiments, so you can use our models to repeat the experiments. If you would like to train the model, such as MNIST (LeNet5) from scratch, you can use the following commands.

```
bash run.sh mnist train
```

#### test

*test* function is used to test the model accuracy. During the test, we use the top-5-accuracy for Tiny ImageNet and top-1-accuracy for others.

*test* function's results are used in Table 1.

```
bash run.sh mnist test
```

#### select

*select* is the most important function in PrivDNN. PrivDNN offers four kinds of algorithms to select critical neurons, i.e., random selection, greedy selection, pruning selection and pruning+greedy selection. All algorithms are listed in **PrivDNN/PrivDNN/main.py**, and you can config which algorithm to use here. For datasets MNIST and EMNIST, we select exact number of neurons in first two layers, for GTSRB, CIFAR10 and Tiny ImageNet, we select a percentage of neurons in first two layers. We also explain parameters in the script.

*select* function's results are used in Table 2, 3, 4, 5 and Figure 4, 7.

```
bash run.sh mnist select
```

#### recover

*recover* function recovers the model with selected neurons or train the model from scratch.

*recover* function's results are used in Table 7 and Figure 7.

```
bash run.sh mnist recover
```

#### inference

*inference* function uses the C++ SEAL library to execute the cipher domain inference. PrivDNN will automatically use the file **PrivDNN/saved_models/[dataset]/inference_encrypted_neurons.json** as the selected neurons. The running time of *inference* function highly depends on the dataset and selected neurons count. During this process, it may take up all CPU cores, generate large cipher files and consume lots of memory (virtual memory). To test the function, you can inference the MNIST dataset with selected neurons of {"1": [0], "2": [0]}, which should finish in five minutes.

*inference* function's results are used in Figure 5, 7.

```
bash run.sh mnist inference
```

#### clean

```
bash run.sh clean
```

*clean* function function deletes the generated cipher parameters. PrivDNN will check if the file exists. If it exists, it will pass the generating process and use it. Otherwise, it will generate required cipher parameters automatically. If you change the SEAL parameters, you must compile and clean encrypted model data to generate new data.

#### others

We have some python programs in **PrivDNN/analyze_result**, to statistic the results we got from previous experiments. The results of those programs are used in Table 2, 3 and Figure 5, 7.

### Citation

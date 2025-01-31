# DS-Pruning: Leveraging Dying Neurons for Sparsity-Aware Pruning in Convolutional Neural Networks
This is the official implementation for DS-Pruning: Leveraging Dying Neurons for Sparsity-Aware Pruning in Convolutional Neural Networks.

# Overview
In this paper, we challenge the conventional view of dying neurons—neurons that cease to activate—during deep neural network training. Traditionally regarded as problematic due to their association with optimization challenges and reduced model adaptability over training epochs, dying neurons are often seen as a hindrance. However, we present a novel perspective, demonstrating that they can be effectively leveraged to enhance network sparsity. Specifically, we propose DS-Pruning, a sparsity-aware pruning approach for convolutional neural networks (CNNs) that exploits the behavior of individual neurons during training. Through a systematic exploration of hyperparameter configurations, we show that dying neurons can be harnessed to improve pruning algorithms. Our method dynamically monitors the occurrence of dying neurons, enabling adaptive sparsification throughout CNN training. Extensive experiments on diverse datasets demonstrate that DS-Pruning outperforms existing sparsity-aware pruning techniques while achieving competitive results compared to state-of-the-art methods. These findings suggest that dying neurons can serve as an efficient mechanism for network compression and resource optimization in CNNs, opening new avenues for more efficient and high-performance deep learning models.

# Results

![BS](https://github.com/wangbst/ExplainableP/assets/97005040/ed999e78-f198-42fb-a556-6f308ac0a163)![LR](https://github.com/wangbst/ExplainableP/assets/97005040/ac4abc77-595f-4d42-9a1f-4e81b2bb2432)![Regularization](https://github.com/wangbst/ExplainableP/assets/97005040/2c054748-7efc-434c-b321-90650f35ded3) 

# Dependencies
```shell
conda create -n myenv python=3.7
conda activate myenv
conda install -c pytorch pytorch==1.9.0 torchvision==0.10.0
pip install scipy
```

# Datasets
Please download the Imagenet Dataset. 

# ResNet18 and Leaky ReLU
All used ResNet18 and Leaky ReLU models can be downloaded from here. Please put them in ResNet18().

# Run dying neurons accumulation for a ResNet-18 trained on CIFAR-10.
 ```shell
$ python Resnet18.py
$ python Leaky ReLU.py
$ python Swish.py
```
- In Leaky ReLU.py, replace activation functions ReLU with LeakyReLU.
- In Swish.py, replace activation functions ReLU with Swish.

# Run SGD noise and SGD for a ResNet-18 trained on CIFAR-10.
 ```shell
$ python SGD noise.py
$ python SGD.py
```

# Run Neural sparsity, structured methods for ResNet-18 on CIFAR-10.
 ```shell
$ python Resnet18.py
$ python Leaky ReLU.py
```

 # Run Weight sparsity, structured methods for ResNet-18 on CIFAR-10.
 ```shell
$ python Resnet18.py
$ python Leaky ReLU.py
```

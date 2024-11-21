# DS-Pruning: Harnessing Dying Neurons for Sparsity-Driven Structured Pruning in Neural Networks
This is the official implementation for Explainable Pruning: Efficient and Explainable Neural Network Pruning Leveraging Neuron Death.

# Overview
We challenge the conventional perspective on "dying neurons" in deep neural network training. Traditionally, this phenomenon has been perceived as undesirable due to its association with optimization difficulties and a decrease in the network's adaptability during continuous learning. However, we propose a novel perspective by examining the relationship between dying neurons and network sparsity, particularly in the context of pruning.Through systematic exploration of various hyperparameter configurations, we uncover the potential of dying neurons to facilitate structured pruning algorithms effectively. Our approach, termed "Explainable pruning," offers a means to regulate the occurrence of dying neurons, dynamically sparsing the neural network during training. Notably, our method is straightforward and widely applicable, surpassing existing structured pruning techniques while achieving results comparable to popular unstructured pruning methods.These findings mark a significant advancement, as they suggest that dying neurons can serve as an efficient model for compressing and optimizing network resources. This insight opens up new avenues for enhancing model efficiency and performance in deep learning applications.

# Results
![1](https://github.com/wangbst/ExplainableP/assets/97005040/d2d1bf37-9494-4b78-ac21-3eadaf3badf0) 

![BS](https://github.com/wangbst/ExplainableP/assets/97005040/ed999e78-f198-42fb-a556-6f308ac0a163)![LR](https://github.com/wangbst/ExplainableP/assets/97005040/ac4abc77-595f-4d42-9a1f-4e81b2bb2432)![Regularization](https://github.com/wangbst/ExplainableP/assets/97005040/2c054748-7efc-434c-b321-90650f35ded3) 

![SGD noise](https://github.com/wangbst/ExplainableP/assets/97005040/9fcbda8e-70dc-457f-a219-ae6afa3599ae) ![SGD](https://github.com/wangbst/ExplainableP/assets/97005040/ee9296ec-fabb-4b5b-83f7-303cca0c35b9)

![2](https://github.com/wangbst/ExplainableP/assets/97005040/f4196e5d-bbe7-4362-a966-9f8235cdb0be) ![3](https://github.com/wangbst/ExplainableP/assets/97005040/bb2f2591-617a-4a28-bd7a-f7ec9393de88) ![4](https://github.com/wangbst/ExplainableP/assets/97005040/eb9e9ee3-f95f-4c00-9c52-ecb43afa40cc) ![5](https://github.com/wangbst/ExplainableP/assets/97005040/c3bd0cfa-7af4-40a4-86a3-c2f92de4d0e8) ![result](https://github.com/wangbst/ExplainableP/assets/97005040/571ae237-939e-49cc-b1b1-c7904e42c73a)

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

# Run ResNet-50 model trained on ImageNet using different criteria when pruning at approximately 80% and 90% weight sparsity.
 ```shell
$ python Imagenet.py
```
Set a new download directory for `'model = torchvision.models.resnet50(pretrained=True)'`, we need to export `'TORCH_HOME=/torch_cache'`

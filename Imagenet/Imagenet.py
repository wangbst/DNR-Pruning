import os
import pathlib
from datetime import datetime
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
import torch.optim as optim
import torch.nn.utils as utils
import math
import torch.nn.utils.prune as prune  # Added import for weight pruning
import torchvision.datasets as datasets
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128

# Path to the compressed archive file
data_path = ''
data_path1 = ''
trainloader, testloader, trainset, testset=load_ImageNet('./')

# Create ResNet50 model
model = torchvision.models.resnet50(pretrained=True)
model = model.to(device)


# Prune 80% of weights across all layers
def prune_weights(model, amount=0.8):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)

# Apply weight pruning
prune_weights(model, amount=0.8)

# Rest of the code remains the same...
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

# Configuration

learning_rate = 0.01
momentum = 0.9
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,weight_decay=5e-4)

indices = [2, 5, 8, 11]
conv_modules = [module for module in model.modules() if isinstance(module, nn.Conv2d)]

entropies = [[] for _ in range(len(conv_modules))]
for idx in indices:
    entropies[idx] = [[] for _ in range(conv_modules[idx].out_channels)]

class ConvEntropyHook:
    def __init__(self, module, idx):
       
    def hook_fn(self, module, input, output):
        
    def close(self):
        self.hook.remove()


hooks = []
for idx in indices:
    hooks.append(ConvEntropyHook(conv_modules[idx], idx))
    

# Create a list to store weight changes for each convolutional layer
weight_changes = [[] for _ in indices]
weight_changes1 = [[] for _ in indices]

# 添加计算平均信息熵的类
class LayerEntropyHook:
    def __init__(self, module, layer_name):
        
    def hook_fn(self, module, input, output):
        # Add a small epsilon to avoid log(0) issues
       
    def close(self):
        self.hook.remove()

layer_names = ['conv1'] + [f'layer{i}' for i in range(1, 5)] + ['fc']
entropy_hooks = [LayerEntropyHook(getattr(model, layer_name), name) for layer_name, name in zip(layer_names, layer_names)]

# Define a function to calculate entropy
def calculate_entropy(tensor):
    epsilon = 1e-10
    tensor = torch.abs(tensor) + epsilon
    entropy = -torch.sum(tensor * torch.log2(tensor), dim=tuple(range(1, tensor.dim())))
    return entropy


val_losses = []
train_losses = []
test_losses = []
pruned_neurons = []  # To store number of pruned neurons per epoch
kl_divergences = []  # To store KL divergence per epoch
best_test_accuracy = 0.0


# Training Loop
for epoch in range(200):  # loop over the dataset
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    pruned_neurons_epoch = 0  # Initialize pruned_neurons_epoch for the current epoch
    kl_divergence = 0.0
    
    model.train()  # Ensure model is in training mode

    for i, data in enumerate(trainloader, 0):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # Use those GPUs!
        inputs, labels = inputs.to(device), labels.to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()
        
        # Calculate accuracy
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()
        train_acc = 100 * correct_train // total_train

        # Print statistics
        running_loss += loss.item()
        
        if i % 10 == 9:
            for idx, layer_idx in enumerate(indices):
                layer = conv_modules[layer_idx]
                weight = layer.weight.data.cpu().numpy()
            
                weight_changes[idx].append([np.mean(weight[i]) for i in range(weight.shape[0])])
                
                weight_change1 = np.mean(weight)
                weight_changes1[idx].append(weight_change1)
        
    # Calculate training error
    train_loss = running_loss / len(trainloader)
    train_losses.append(train_loss)
    
    print(f'Training Error at Epoch {epoch + 1}: {train_loss}')
   
    model.eval()
    val_loss = 0.0
    correct_test = 0
    total_test = 0
    
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Calculate accuracy
            _, predicted_test = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted_test == labels).sum().item()
            
    # Calculate test error
    test_loss = val_loss / len(testloader.dataset)
    test_acc = round(100 * correct_test / total_test, 2)
    test_losses.append(test_loss)

    print(f'Test Error at Epoch {epoch + 1}: {test_loss}')
    print(f'Test Accuracy at Epoch {epoch + 1}: {test_acc}%')
    
    # Update best test accuracy
    if test_acc > best_test_accuracy:
        best_test_accuracy = test_acc
    
    model.train()

    val_losses.append(test_loss)

    # Perform pruning based on KL divergence and entropy reduction
    for idx, layer_idx in enumerate(indices):
        for ch_idx, entropies_ch in enumerate(entropies[layer_idx]):
            prev_entropy = entropies[layer_idx][ch_idx][epoch - 1] if epoch > 0 else entropies[layer_idx][ch_idx][0]
            curr_entropy = entropies[layer_idx][ch_idx][epoch]
            entropy_reduction = prev_entropy - curr_entropy
            
            if entropy_reduction < 0:  
                kl_divergence += abs(entropy_reduction)
            
            if entropy_reduction < 0 and kl_divergences[epoch] - kl_divergences[epoch - 1] > 0:  # If entropy reduced (bad) and KL divergence is non-zero
                
                layer = conv_modules[layer_idx]
                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias.data[ch_idx].zero_()
                layer.weight.data[ch_idx].zero_()
                    
                pruned_neurons_epoch += 1
                
    kl_divergences.append(kl_divergence)
    pruned_neurons.append(pruned_neurons_epoch)
    
    # Reset pruned_neurons_epoch and kl_divergence after logging
    pruned_neurons_epoch = 0
    kl_divergence = 0.0

# Close all hooks
for hook in hooks:
    hook.close()

for hook in entropy_hooks:
    hook.close()

# Print best test accuracy
print(f"Best Test Accuracy: {best_test_accuracy}%")
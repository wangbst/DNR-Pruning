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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

batch_size = 16

data_path = 'data'

trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = 10

# Define Resnet using default weights
net_1 = torchvision.models.resnet18()
net_1.name = "resnet18"

# Replace ReLU with LeakyReLU
for module in net_1.modules():
    if isinstance(module, nn.ReLU):
        module = nn.LeakyReLU(inplace=True)

# Set the model to run on the device
net_1 = net_1.to(device)

import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

# Configuration

model = net_1
learning_rate = 0.05  #0.01
momentum = 0.9
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

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
    
        # Calculate entropy and append to the list
        
    def close(self):
        self.hook.remove()

# 添加对输入层和各卷积层的平均信息熵的计算
layer_names = ['conv1'] + [f'layer{i}' for i in range(1, 5)] + ['fc']
# Instantiate LayerEntropyHook for each layer
entropy_hooks = [LayerEntropyHook(getattr(model, layer_name), name) for layer_name, name in zip(layer_names, layer_names)]


# Define a function to calculate entropy
def calculate_entropy(tensor):
    epsilon = 1e-10
    tensor = torch.abs(tensor) + epsilon
    entropy = -torch.sum(tensor * torch.log2(tensor), dim=tuple(range(1, tensor.dim())))
    return entropy


run = wandb.init(project="6", 
                 config={"batch_size": batch_size,
                         "learning_rate": learning_rate,
                         "momentum": momentum,
                         "model": model.name} )

val_losses = []
train_losses = []
test_losses = []
pruned_neurons = []  # To store number of pruned neurons per epoch
kl_divergences = []  # To store KL divergence per epoch

# Training Loop
for epoch in range(100):  # loop over the dataset
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    pruned_neurons_epoch = 0  # Initialize pruned_neurons_epoch for the current epoch
    kl_divergence = 0.0

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
        
        # Gradient Clipping
        utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
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
    
    # 记录信息熵到WandB
    wandb.log({'epoch': epoch, 'accuracy': train_acc, 'loss': train_loss})
    
    # 记录各层信息熵到WandB
    for hook in entropy_hooks:
        wandb.log({f'{hook.layer_name}_entropy': torch.mean(torch.tensor(hook.entropies))})
        
    print(f'Training Error at Epoch {epoch + 1}: {train_loss}')
        
    model.eval()
    val_loss = 0.0
    correct_test = 0
    total_test = 0
    
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
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
    test_acc = 100 * correct_test // total_test
    test_losses.append(test_loss)

    print(f'Test Error at Epoch {epoch + 1}: {test_loss}')
    print(f'Test Accuracy at Epoch {epoch + 1}: {test_acc}%')
    
    model.train()

    # Log test error
    wandb.log({'epoch': epoch, 'val_loss': test_loss, 'val_accuracy': test_acc})
    val_losses.append(test_loss)

    # Perform pruning based on KL divergence and entropy reduction
    for idx, layer_idx in enumerate(indices):
        for ch_idx, entropies_ch in enumerate(entropies[layer_idx]):
            prev_entropy = entropies[layer_idx][ch_idx][epoch - 1] if epoch > 0 else entropies[layer_idx][ch_idx][0]
            curr_entropy = entropies[layer_idx][ch_idx][epoch]
            entropy_reduction = prev_entropy - curr_entropy
            
            if entropy_reduction < 0:  # If entropy reduced (bad)
                kl_divergence += abs(entropy_reduction)
            
            if entropy_reduction < 0 and kl_divergences[epoch]-- kl_divergences[epoch - 1] > 0:  # If entropy reduced (bad) and KL divergence is non-zero
                
                layer = conv_modules[layer_idx]
                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias.data[ch_idx].zero_()
                layer.weight.data[ch_idx].zero_()
                    
                pruned_neurons_epoch += 1
                
    pruned_neurons.append(pruned_neurons_epoch)
    kl_divergences.append(kl_divergence)
    
    # Logging pruned neurons and KL divergence on WandB
    wandb.log({'epoch': epoch, 'pruned_neurons': pruned_neurons_epoch, 'kl_divergence': kl_divergence})
    
    # Reset pruned_neurons_epoch and kl_divergence after logging
    pruned_neurons_epoch = 0
    kl_divergence = 0.0

# Close all hooks
for hook in hooks:
    hook.close()

for hook in entropy_hooks:
    hook.close()

# Close WandB run
wandb.finish()

plot_interval = 2
plt.figure(figsize=(30, 20), dpi=300)
num_layers = len(indices)
for i, idx in enumerate(indices):
    plt.subplot(num_layers//2, 2, i+1)
    for ch in range(conv_modules[idx].out_channels):
        plt.plot(range(0, len(entropies[idx][ch]), plot_interval), entropies[idx][ch][::plot_interval], color="C0", alpha=0.075)
    plt.title(f'Layer: {idx}')
    plt.xlabel('steps')
    plt.ylabel('float entropy')
    
    ax = plt.gca()
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.savefig('entropy_plot (Leaky ReLU).png')
plt.show()
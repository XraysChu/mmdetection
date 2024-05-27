import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms
# Now you can use this dataset with a DataLoader to iterate over it in batches
from torch.utils.data import DataLoader

epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = "data/lenet_mnist_model.pth"
use_cuda=True
# Set random seed for reproducibility
torch.manual_seed(42)


# Model definition
cfg = Config.fromfile('/home/exouser/mmdetection/workspace/20240405_073629/vis_data/config.py')
runner = Runner.from_cfg(cfg)
model = runner.model


# kitti Test dataset and dataloader declaration
# Load the KITTI test dataset
# cfg.data.test.type = 'CocoDataset'
# cfg.data.test.data_root = '/home/exouser/data/kitti'
# cfg.data.test.ann_file = '/home/exouser/data/kitti/annotations/instances_val2017.json'
# cfg.data.test.img_prefix = '/home/exouser/data/kitti/val2017'


# test_dataset = YourTestDataset()  # Replace YourTestDataset with the actual test dataset
# test_dataloader = torch.utils.data.DataLoader(
#     test_dataset,
#     batch_size=1,
#     shuffle=False,
#     num_workers=1,
#     collate_fn=test_dataset.collate
# )

# Path to your COCO data and annotations
img_folder = "/home/exouser/data/kitti/val2017"
ann_file = "/home/exouser/data/kitti/annotations/instances_val2017.json"

# Define the transform to apply to the images
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
    # Add any other transformations you need
])

# Load the dataset
dataset = CocoDetection(img_folder, ann_file, transform=transform)

test_loader = DataLoader(dataset, batch_size=1, shuffle=True)


# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

# Initialize the network
model = model.to(device)

# Load the pretrained model
weights = '/home/exouser/mmdetection/workspace/epoch_12.pth'
state_dict = torch.load(weights)
model.load_state_dict(state_dict['state_dict'])

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

# restores the tensors to their original scale
def denorm(batch, mean=[0.1307], std=[0.3081]):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def test( model, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        # print(f'data: {data}')
        # print(f'target: {target}')
        # Extract bounding boxes and labels from target
        # boxes = [item['bbox'] for item in target]
        # labels = [item['category_id'] for item in target]
        # # Convert lists to tensors
        # boxes = torch.stack(boxes).to(device)
        # labels = torch.stack(labels).to(device)
        # Extract bounding boxes and labels from target
        boxes = [torch.cat(item['bbox']).float() for item in target]
        labels = [item['category_id'] for item in target]

        # Convert lists to tensors
        boxes = torch.stack(boxes).to(device)
        labels = torch.stack(labels).to(device)
        data = torch.tensor(data).to(device)
        # data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        # Check if output is None
        if output is None:
            init_pred = torch.tensor([0])
            print("Model did not return any output")
        else:
            init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, don't bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect ``datagrad``
        data_grad = data.grad.data

        # Restore the data to its original scale
        data_denorm = denorm(data)

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)

        # Reapply normalization
        perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)

        # Re-classify the perturbed image
        output = model(perturbed_data_normalized)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)


plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()

# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel(f"Eps: {epsilons[i]}", fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title(f"{orig} -> {adv}")
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()
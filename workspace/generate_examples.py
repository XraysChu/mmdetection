import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Clip the perturbed image to ensure it stays within the valid range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def generate_adversarial_examples(model, images, labels, epsilon):
    # Set the model to evaluation mode
    model.eval()
    # Enable gradient calculation
    images.requires_grad = True
    # Forward pass
    outputs = model(images)
    # Calculate the loss
    loss = nn.CrossEntropyLoss()(outputs, labels)
    # Zero the gradients
    model.zero_grad()
    # Backward pass
    loss.backward()
    # Collect the gradients of the input image
    data_grad = images.grad.data
    # Generate adversarial examples using FGSM
    perturbed_images = fgsm_attack(images, epsilon, data_grad)
    # print(perturbed_images)
    # Return the perturbed images
    # print(f'outputs: {outputs}, labels:{labels}')
    return perturbed_images

# # Example usage
# # Create an instance of the Config class
# cfg = Config.fromfile('/home/exouser/mmdetection/workspace/20240405_073629/vis_data/config.py')

# # Build your model using the config
# runner = Runner.from_cfg(cfg)

# model = runner.model

# Example code using PyTorch and torchvision
import torch
from torchvision import models, transforms
from PIL import Image
import os
import shutil
import json

# Load pre-trained ResNet model
resnet_model = models.resnet50(pretrained=True)
# resnet_model.eval()

# Replace the placeholder with the code above
# model = YourModel()  # Replace with your own model
# Define the paths
input_dir = '/home/exouser/data/kitti/val2017'
output_dir = '/home/exouser/data/kitti/adver_example_005/'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

epsilon = 0.007843137255  # Adjust the epsilon value as desired
epsilon = 0.05

# Iterate over the input images
# Load the labels from the JSON file
# with open('/home/exouser/data/kitti/annotations/instances_val2017.json', 'r') as f:
#     data = json.load(f)
#     labels = data['annotations']
#     # Convert labels to tensor
#     labels = torch.tensor(labels)
# Iterate over the input images
for filename in os.listdir(input_dir):
    # Load the image
    image_path = os.path.join(input_dir, filename)
    image = Image.open(image_path)

    # Preprocess the image
    # preprocess = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    # input_image = preprocess(image).unsqueeze(0)
    input_image = transforms.ToTensor()(image).unsqueeze(0)

    # Generate random labels
    labels = torch.randint(0, 3, (1,))

    # Generate the adversarial example
    perturbed_image = generate_adversarial_examples(resnet_model, input_image, labels, epsilon)

    # Save the perturbed image
    output_path = os.path.join(output_dir, filename)
    perturbed_image = perturbed_image.squeeze(0).detach().cpu()
    perturbed_image = transforms.ToPILImage()(perturbed_image)
    perturbed_image = perturbed_image.resize(image.size)
    perturbed_image.save(output_path)

    # Print the progress
    print(f'Saved adversarial example: {output_path}')
# images = torch.randn(10, 3, 224, 224)  # Replace with your own input images
# labels = torch.tensor([0, 1, 2,0, 1, 2,0, 1, 2, 0])  # Replace with your own labels


# perturbed_images = generate_adversarial_examples(resnet_model, images, labels, epsilon)
Instructions for the two functions get_data and show_batch_image:

# Neural Network Helper Functions

This repository contains two helper functions for working with neural networks using PyTorch. These functions simplify the process of loading data and visualizing batch images during model training.

## Function: get_data

This function is used to load the MNIST dataset for training and testing a neural network model.

### Usage

```python
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    # Add any additional transformations you require
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    # Add any additional transformations you require
])

train_data, test_data = get_data(train_transforms, test_transforms)



## Function: show_batch_image
This function allows you to visualize a batch of images from the training dataset.

### Usage
show_batch_image(train_loader)

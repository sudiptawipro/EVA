from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
from torchsummary import summary


def get_data(train_transforms,test_transforms):
  train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
  test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)
  return train_data,test_data




def show_batch_image(train_loader):
  batch_data, batch_label = next(iter(train_loader)) 

  fig = plt.figure()

  for i in range(12):
    plt.subplot(3,4,i+1)
    plt.tight_layout()
    plt.imshow(batch_data[i].squeeze(0), cmap='gray')
    plt.title(batch_label[i].item())
    plt.xticks([])
    plt.yticks([])


def train_test_acc(train_losses,train_acc,test_losses,test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")


def model_summary(Net):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net().to(device)
    summary(model, input_size=(1, 28, 28))
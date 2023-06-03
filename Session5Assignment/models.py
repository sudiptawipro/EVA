{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU",
    "gpuClass": "standard"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/sudiptawipro/EVA/blob/main/Session5Assignment/models.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "CODE BLOCK: 1"
      ],
      "metadata": {
        "id": "n09vaEgP6pLj"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ZBDVm8pSrhbr",
        "outputId": "98bea8ab-f9f6-4728-b245-5346261670c3"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "6PlbomWY3RSq",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "927e706e-5520-42e0-f5cb-a22524285c90"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/\n",
            "Requirement already satisfied: torchsummary in /usr/local/lib/python3.10/dist-packages (1.5.1)\n"
          ]
        }
      ],
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "import torch.optim as optim\n",
        "from torchvision import datasets, transforms\n",
        "import matplotlib.pyplot as plt\n",
        "import utils\n",
        "!pip install torchsummary"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "CODE BLOCK: 2"
      ],
      "metadata": {
        "id": "VjBHHQVA6sXt"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# CUDA?\n",
        "cuda = torch.cuda.is_available()\n",
        "print(\"CUDA Available?\", cuda)\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "94BxVVBP3WwS",
        "outputId": "88ffea66-59b5-4062-c40f-0a5ece942815"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "CUDA Available? True\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "CODE BLOCK: 3"
      ],
      "metadata": {
        "id": "3UHq59Sw6tmW"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Train data transformations\n",
        "train_transforms = transforms.Compose([\n",
        "    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),\n",
        "    transforms.Resize((28, 28)),\n",
        "    transforms.RandomRotation((-15., 15.), fill=0),\n",
        "    transforms.ToTensor(),\n",
        "    transforms.Normalize((0.1307,), (0.3081,)),\n",
        "    ])\n",
        "\n",
        "# Test data transformations\n",
        "test_transforms = transforms.Compose([\n",
        "    transforms.ToTensor(),\n",
        "    transforms.Normalize((0.1407,), (0.4081,))\n",
        "    ])"
      ],
      "metadata": {
        "id": "KpshQ2Ug38m2"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "CODE BLOCK: 4"
      ],
      "metadata": {
        "id": "zQm17pM46zHL"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "\n",
        "import sys  \n",
        "  \n",
        "path_to_util = '/content/utils.py'\n",
        "path_to_model = '/content/models.py'\n",
        "\n",
        "from utils import get_data\n",
        "train_data, test_data = get_data(train_transforms,test_transforms)\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "JB79ZYW13-AO"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "CODE BLOCK: 5"
      ],
      "metadata": {
        "id": "_PKSHxto6116"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "batch_size = 512\n",
        "\n",
        "kwargs = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 2, 'pin_memory': True}\n",
        "\n",
        "test_loader = torch.utils.data.DataLoader(train_data, **kwargs)\n",
        "train_loader = torch.utils.data.DataLoader(train_data, **kwargs)"
      ],
      "metadata": {
        "id": "avCKK1uL4A68"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "CODE BLOCK: 6"
      ],
      "metadata": {
        "id": "Hi_0rfq56-29"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#import matplotlib.pyplot as plt\n",
        "\n",
        "#batch_data, batch_label = next(iter(train_loader)) \n",
        "\n",
        "#fig = plt.figure()\n",
        "\n",
        "#for i in range(12):\n",
        "  #plt.subplot(3,4,i+1)\n",
        "  #plt.tight_layout()\n",
        "  #plt.imshow(batch_data[i].squeeze(0), cmap='gray')\n",
        "  #plt.title(batch_label[i].item())\n",
        "  #plt.xticks([])\n",
        "  #plt.yticks([])\n",
        "\n",
        "from utils import show_batch_image\n",
        "show_batch_image(train_loader)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 463
        },
        "id": "Hx7QkLcw4Epc",
        "outputId": "31ae04dd-9cf1-4ba6-d5a7-9b1ad27336e6"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 12 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "CODE BLOCK: 7"
      ],
      "metadata": {
        "id": "y3XbVJSb7Aee"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# class Net(nn.Module):\n",
        "#     #This defines the structure of the NN.\n",
        "#     def __init__(self):\n",
        "#         super(Net, self).__init__()\n",
        "#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3)\n",
        "#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3)\n",
        "#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3)\n",
        "#         self.conv4 = nn.Conv2d(128, 256, kernel_size=3)\n",
        "#         self.fc1 = nn.Linear(320, 50)\n",
        "#         self.fc2 = nn.Linear(50, 10)\n",
        "\n",
        "#     def forward(self, x):\n",
        "#         x = F.relu(self.conv1(x), 2)\n",
        "#         x = F.relu(F.max_pool2d(self.conv2(x), 2)) \n",
        "#         x = F.relu(self.conv3(x), 2)\n",
        "#         x = F.relu(F.max_pool2d(self.conv4(x), 2)) \n",
        "#         x = x.view(-1, 320)\n",
        "#         x = F.relu(self.fc1(x))\n",
        "#         x = self.fc2(x)\n",
        "#         return F.log_softmax(x, dim=1)\n",
        "\n",
        "path_to_model = '/content/models.py'\n",
        "from models import Net\n"
      ],
      "metadata": {
        "id": "UHBolvMH4F8y"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "CODE BLOCK: 8"
      ],
      "metadata": {
        "id": "89gd4_s7AO2y"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Data to plot accuracy and loss graphs\n",
        "train_losses = []\n",
        "test_losses = []\n",
        "train_acc = []\n",
        "test_acc = []\n",
        "\n",
        "test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}"
      ],
      "metadata": {
        "id": "7du4zM474LvT"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "CODE BLOCK: 9"
      ],
      "metadata": {
        "id": "kCwIPHqwAQgB"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "from tqdm import tqdm\n",
        "\n",
        "def GetCorrectPredCount(pPrediction, pLabels):\n",
        "  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()\n",
        "\n",
        "def train(model, device, train_loader, optimizer):\n",
        "  model.train()\n",
        "  pbar = tqdm(train_loader)\n",
        "\n",
        "  train_loss = 0\n",
        "  correct = 0\n",
        "  processed = 0\n",
        "\n",
        "  for batch_idx, (data, target) in enumerate(pbar):\n",
        "    data, target = data.to(device), target.to(device)\n",
        "    optimizer.zero_grad()\n",
        "\n",
        "    # Predict\n",
        "    pred = model(data)\n",
        "\n",
        "    # Calculate loss\n",
        "    loss = F.nll_loss(pred, target)\n",
        "    train_loss+=loss.item()\n",
        "\n",
        "    # Backpropagation\n",
        "    loss.backward()\n",
        "    optimizer.step()\n",
        "    \n",
        "    correct += GetCorrectPredCount(pred, target)\n",
        "    processed += len(data)\n",
        "\n",
        "    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')\n",
        "\n",
        "  train_acc.append(100*correct/processed)\n",
        "  train_losses.append(train_loss/len(train_loader))\n",
        "\n",
        "def test(model, device, test_loader):\n",
        "    model.eval()\n",
        "\n",
        "    test_loss = 0\n",
        "    correct = 0\n",
        "\n",
        "    with torch.no_grad():\n",
        "        for batch_idx, (data, target) in enumerate(test_loader):\n",
        "            data, target = data.to(device), target.to(device)\n",
        "\n",
        "            output = model(data)\n",
        "            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss\n",
        "\n",
        "            correct += GetCorrectPredCount(output, target)\n",
        "\n",
        "\n",
        "    test_loss /= len(test_loader.dataset)\n",
        "    test_acc.append(100. * correct / len(test_loader.dataset))\n",
        "    test_losses.append(test_loss)\n",
        "\n",
        "    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\\n'.format(\n",
        "        test_loss, correct, len(test_loader.dataset),\n",
        "        100. * correct / len(test_loader.dataset)))\n",
        "     "
      ],
      "metadata": {
        "id": "gpNw3-sy4QGd"
      },
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "CODE BLOCK: 10"
      ],
      "metadata": {
        "id": "09GYKBGRAT5M"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from models import Net\n",
        "\n",
        "model = Net().to(device)\n",
        "optimizer = optim.SGD(model.parameters(), lr=.01, momentum=0.9)\n",
        "scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)\n",
        "num_epochs = 1\n",
        "\n",
        "for epoch in range(1, num_epochs+1):\n",
        "  print(f'Epoch {epoch}')\n",
        "  train(model, device, train_loader, optimizer)\n",
        "  test(model, device, train_loader)\n",
        "  scheduler.step()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Owqiet9M4TV7",
        "outputId": "74ffe6d8-e68f-45a9-9aeb-f61d74a55a93"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Adjusting learning rate of group 0 to 1.0000e-02.\n",
            "Epoch 1\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Train: Loss=0.4148 Batch_id=117 Accuracy=48.38: 100%|██████████| 118/118 [00:26<00:00,  4.45it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Test set: Average loss: 0.3925, Accuracy: 52971/60000 (88.28%)\n",
            "\n",
            "Adjusting learning rate of group 0 to 1.0000e-02.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "CODE BLOCK: 11"
      ],
      "metadata": {
        "id": "B-LM-Z1k6FcF"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# fig, axs = plt.subplots(2,2,figsize=(15,10))\n",
        "# axs[0, 0].plot(train_losses)\n",
        "# axs[0, 0].set_title(\"Training Loss\")\n",
        "# axs[1, 0].plot(train_acc)\n",
        "# axs[1, 0].set_title(\"Training Accuracy\")\n",
        "# axs[0, 1].plot(test_losses)\n",
        "# axs[0, 1].set_title(\"Test Loss\")\n",
        "# axs[1, 1].plot(test_acc)\n",
        "# axs[1, 1].set_title(\"Test Accuracy\")\n",
        "\n",
        "from utils import train_test_acc\n",
        "train_test_acc(train_losses,train_acc,test_losses,test_acc)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 444
        },
        "id": "Wu0l7dli4eC9",
        "outputId": "1a2772bd-96a4-4980-8da4-8e6ece8f9805"
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1500x1000 with 4 Axes>"
            ],
            "image/png": "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GjdurIoVK2rVqlWSyi7HWb58uZo0aaJ77rnHbKtYsaIGDBigAwcO6Oeffzbnd+TIkQJLh/yZt7e3NmzYoJSUlBKfN4CSRfEMQImpWrWqXF1dC7Tv2LFDDz74oLy8vOTp6SlfX19zEdmMjIzLjlu9enWHzxcLaVYFpkv1vdj/Yt9jx47p7NmzqlWrVoG4wtquxMGDByVJtWvXLrCvTp065n673a433nhDX3zxhfz9/dWiRQuNHz9eaWlpZnzLli3VpUsXjR07VlWqVFHHjh01a9Ys5eTkXJO5AgAAXEu7d+9WRkaG/Pz85Ovr67BlZWXp2LFjksouxzl48GChOdrFt4NezNNGjBihihUrqkmTJrrllls0ePBgfffddw59xo8fr+3btys4OFhNmjTRmDFjivRjL4DrD8UzACXmz3eYXXTq1Cm1bNlSP/30k8aNG6dly5Zp5cqVeuONNyRdWED2cpydnQttNwyjRPuWhWeeeUa//vqr4uPj5ebmppdffll169bVjz/+KOnCSxA+/vhjJSUlaciQITp69Kj69u2rxo0bKysrq4xnDwAA4Cg/P19+fn5auXJlodu4ceMkXf85Tt26dbVr1y4tWLBA99xzjz755BPdc889Gj16tBnTrVs37du3T++8846CgoL05ptvql69evriiy/KcOYArgTFMwClavXq1Tp58qRmz56tp59+Wv/85z8VFRXl8BhmWfLz85Obm5v27NlTYF9hbVciJCREkrRr164C+3bt2mXuvygsLEzPPvusvvrqK23fvl25ubn697//7RBz11136bXXXtOmTZuUkJCgHTt2aMGCBddkvgAAANdKWFiYTp48qWbNmikqKqrA1rBhQ4f4S+U41/It6BeFhIQUmqNdXFrkz3lahQoV1L17d82aNUuHDh1S+/bt9dprr+ncuXNmTGBgoAYNGqQlS5Zo//79qly5sl577bVrPm8AJYviGYBSdfHOrz/f6ZWbm6upU6eW1ZQcODs7KyoqSkuWLHFYn2LPnj3X7FfCiIgI+fn5afr06Q6PHnzxxRf65Zdf1L59e0nSmTNnHJIv6ULC6eHhYfb7/fffC9w1Fx4eLkk8ugkAAK473bp1U15enl555ZUC+/744w+dOnVKUtFynPLly0uS2edauP/++7Vx40YlJSWZbdnZ2Xr//fcVGhqq2267TdKFt8r/maurq2677TYZhqHz588rLy+vwHIkfn5+CgoKIkcDbkAuZT0BADeXu+++W5UqVVLv3r311FNPyWazad68edfVY5NjxozRV199pWbNmunJJ59UXl6e3n33XdWvX1/JyclFGuP8+fN69dVXC7T7+Pho0KBBeuONNxQTE6OWLVuqZ8+eSk9P16RJkxQaGqphw4ZJkn799Vfde++96tatm2677Ta5uLjo008/VXp6unr06CFJmjNnjqZOnaoHH3xQYWFhOn36tD744AN5enrq/vvvv2bXBAAA4Fpo2bKlBg4cqPj4eCUnJ6tNmzYqV66cdu/erUWLFmnSpEnq2rVrkXIcd3d33XbbbVq4cKFuvfVW+fj4qH79+qpfv/4l5/DJJ58U+pKq3r17a+TIkfroo4/Url07PfXUU/Lx8dGcOXO0f/9+ffLJJ3JyunD/SZs2bRQQEKBmzZrJ399fv/zyi9599121b99eHh4eOnXqlKpVq6auXbuqYcOGqlixor7++mv98MMPBZ4gAHD9o3gGoFRVrlxZn3/+uZ599lm99NJLqlSpkh599FHde++9io6OLuvpSZIaN26sL774Qs8995xefvllBQcHa9y4cfrll1+K9DZQ6cLddC+//HKB9rCwMA0aNEh9+vRR+fLl9frrr2vEiBGqUKGCHnzwQb3xxhvm26WCg4PVs2dPJSYmat68eXJxcVGdOnX0n//8R126dJF0IQHduHGjFixYoPT0dHl5ealJkyZKSEhQjRo1rtk1AQAAuFamT5+uxo0b67333tMLL7wgFxcXhYaG6tFHH1WzZs0kFT3H+b//+z8NHTpUw4YNU25urkaPHn3Z4pnV0hatWrXSPffco++//14jRozQO++8o3PnzqlBgwZatmyZ+XSAJA0cOFAJCQmaMGGCsrKyVK1aNT311FN66aWXJF24K27QoEH66quvtHjxYuXn56tWrVqaOnWqnnzyyau9hABKmc24nm73AIDrWKdOnbRjxw7t3r27rKcCAAAAACglrHkGAIU4e/asw+fdu3dr+fLlatWqVdlMCAAAAABQJrjzDAAKERgYqD59+qhmzZo6ePCgpk2bppycHP3444+65ZZbynp6AAAAAIBSwppnAFCItm3b6qOPPlJaWprsdruaNm2qf/3rXxTOAAAAAOAmw51nAAAAAAAAgAXWPAMAAAAAAAAsUDwDAAAA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r7RyRUk6ceKEfH19NXr0aI0ZM6ZYfR955BF9//33OnDggFauXKmoqKirmgsAFJVLWU8AQMl79NFHHT6vX79eK1euLND+V2fOnFH58uWLfJxy5cpd0fwkycXFRS4uZfufpE8++UR5eXmaOXOm/vGPf2jt2rVq2bJlmc6pMIZh6Ny5c3J3dy/rqQAAgL+BK80VS1N2drY+++wzxcfHa9asWUpISLhui2fZ2dmqUKFCWU8DwDXEY5sAJEmtWrVS/fr1tXnzZrVo0ULly5fXCy+8IEn67LPP1L59ewUFBclutyssLEyvvPKK8vLyHMb465pnBw4ckM1m01tvvaX3339fYWFhstvtuvPOO/XDDz849C1szTObzaYhQ4ZoyZIlql+/vux2u+rVq6cVK1YUmP/q1asVEREhNzc3hYWF6b333iv2OmoJCQm677771Lp1a9WtW1cJCQmFxu3cuVPdunWTr6+v3N3dVbt2bb344osOMUePHlW/fv3Ma1ajRg09+eSTys3NtTxfSZo9e7ZsNpsOHDhgtoWGhuqf//ynvvzyS0VERMjd3V3vvfeeJGnWrFn6xz/+IT8/P9ntdt12222aNm1aofP+4osv1LJlS3l4eMjT01N33nmn5s+fL0kaPXq0ypUrp+PHjxfoN2DAAHl7e+vcuXOXv4gAAOBvKT8/XxMnTlS9evXk5uYmf39/DRw4UL///rtD3KZNmxQdHa0qVarI3d1dNWrUUN++fSVdyA19fX0lSWPHjjUfBy3KHWiffvqpzp49q4ceekg9evTQ4sWLC81Nzp07pzFjxujWW2+Vm5ubAgMD1blzZ+3du9fhXCZNmqTbb79dbm5u8vX1Vdu2bbVp0yZznjabTbNnzy4w/l/nezGn+/nnn/Xwww+rUqVKuueeeyRJW7duVZ8+fVSzZk25ubkpICBAffv21cmTJwuMe6nccd++fbLZbHr77bcL9Pv+++9ls9n00UcfXfYaArhy3HkGwHTy5Em1a9dOPXr00KOPPmrelj979mxVrFhRsbGxqlixor755huNGjVKmZmZevPNNy877vz583X69GkNHDhQNptN48ePV+fOnbVv377L3q22bt06LV68WIMGDZKHh4cmT56sLl266NChQ6pcubIk6ccff1Tbtm0VGBiosWPHKi8vT+PGjTOTs6JISUnRqlWrNGfOHElSz5499fbbb+vdd9+Vq6urGbd161Y1b95c5cqV04ABAxQaGqq9e/dq2bJleu2118yxmjRpolOnTmnAgAGqU6eOjh49qo8//lhnzpxxGK+odu3apZ49e2rgwIHq37+/ateuLUmaNm2a6tWrpwceeEAuLi5atmyZBg0apPz8fA0ePNjsP3v2bPXt21f16tVTXFycvL299eOPP2rFihV6+OGH9dhjj2ncuHFauHChhgwZYvbLzc3Vxx9/rC5dupTpI7UAAKBsDRw4ULNnz1ZMTIyeeuop7d+/X++++65+/PFHfffddypXrpyOHTumNm3ayNfXVyNHjpS3t7cOHDigxYsXS5J8fX01bdo0Pfnkk3rwwQfVuXNnSVKDBg0ue/yEhAS1bt1aAQEB6tGjh0aOHKlly5bpoYceMmPy8vL0z3/+U4mJierRo4eefvppnT59WitXrtT27dsVFhYmSerXr59mz56tdu3a6fHHH9cff/yhb7/9VuvXr1dERMQVXZ+HHnpIt9xyi/71r3/p4spIK1eu1L59+xQTE6OAgADt2LFD77//vnbs2KH169ebP6ReLnesWbOmmjVrpoSEBA0bNqzAdfHw8FDHjh2vaN4AisgAcNMZPHiw8dd//i1btjQkGdOnTy8Qf+bMmQJtAwcONMqXL2+cO3fObOvdu7cREhJift6/f78hyahcubLx22+/me2fffaZIclYtmyZ2TZ69OgCc5JkuLq6Gnv27DHbfvrpJ0OS8c4775htHTp0MMqXL28cPXrUbNu9e7fh4uJSYEwrb731luHu7m5kZmYahmEYv/76qyHJ+PTTTx3iWrRoYXh4eBgHDx50aM/Pzzf/3KtXL8PJycn44YcfChznYlxh52sYhjFr1ixDkrF//36zLSQkxJBkrFixokB8Yd9NdHS0UbNmTfPzqVOnDA8PDyMyMtI4e/as5bybNm1qREZGOuxfvHixIclYtWpVgeMAAIC/p7/mit9++60hyUhISHCIW7FihUP7p59+akgqNAe66Pjx44YkY/To0UWeT3p6uuHi4mJ88MEHZtvdd99tdOzY0SFu5syZhiRjwoQJBca4mPN88803hiTjqaeesoy5mMPOmjWrQMxf534xp+vZs2eB2MLytI8++siQZKxdu9ZsK0ru+N577xmSjF9++cXcl5uba1SpUsXo3bt3gX4Ari0e2wRgstvtiomJKdD+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3bpKTU3VokWLtG7dOnl7e6tNmzaqXr26+vXrp+HDh8vZ2VkzZ86Ur69vgcKclTZt2sjV1VUdOnTQwIEDlZWVpQ8++EB+fn5KTU014zw9PfX222/r8ccf15133qmHH35YlSpV0k8//aQzZ844FOzKlSunHj166N1335Wzs7N69uxZpLkAAIC/r5YtW2rgwIGKj49XcnKy2rRpo3Llymn37t1atGiRJk2apK5du2rOnDmaOnWqHnzwQYWFhen06dP64IMP5Onpqfvvv1/Shccbb7vtNi1cuFC33nqrfHx8VL9+/ULX/NqwYYP27NmjIUOGFDqvqlWr6o477lBCQoJGjBihXr16ae7cuYqNjdXGjRvVvHlzZWdn6+uvv9agQYPUsWNHtW7dWo899pgmT56s3bt3m49Qfvvtt2rdurV5rMcff1yvv/66Hn/8cUVERGjt2rX69ddfi3zNPD091aJFC40fP17nz59X1apV9dVXXxX69EZRcseLevXqpcmTJ2vVqlV64403ijwfAFepDN7wCaCM/fX144ZhGC1btjTq1atXaPx3331n3HXXXYa7u7sRFBRkPP/888aXX35pSDJWrVplxvXu3dsICQkxP198zfebb75ZYExZvOb7rzGDBw8u0DckJKTAK7kTExONRo0aGa6urkZYWJjxf//3f8azzz5ruLm5WVyFC6/3rly5stG8eXPLGMMwjBo1ahiNGjUyP2/fvt148MEHDW9vb8PNzc2oXbu28fLLLzv0OXjwoNGrVy/D19fXsNvtRs2aNY3BgwcbOTk5ZszmzZuNyMhIw9XV1ahevboxYcIEY9asWYYkY//+/Q7n2759+0LntnTpUqNBgwaGm5ubERoaarzxxhvma9r/PMbF2Lvvvttwd3c3PD09jSZNmhgfffRRgTE3btxoSDLatGlzyesCAAD+ngrLFQ3DMN5//32jcePGhru7u+Hh4WHcfvvtxvPPP2+kpKQYhmEYW7ZsMXr27GlUr17dsNvthp+fn/HPf/7T2LRpk8M433//vdG4cWPD1dW1QE74Z0OHDjUkGXv37rWc65gxYwxJxk8//WQYhmGcOXPGePHFF40aNWoY5cqVMwICAoyuXbs6jPHHH38Yb775plGnTh3D1dXV8PX1Ndq1a2ds3rzZjDlz5ozRr18/w8vLy/Dw8DC6detmHDt2zDKHPX78eIG5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+OEYOnRo9OjRIyorK2PhwoXZeZlMJp555pno169fHqsFAPiT0nwXAADAru3hhx+OJEnioIMOijfeeCMmTZoUvXr1itGjR0dRUVFMmDAhfvrTn8YBBxwQPXr0iGnTpkVVVVV8//vfz3fpAADCMwAAcquuri6mTp0a7733Xuy5555x6qmnxsyZM6NVq1YREfFP//RP8emnn8aYMWNiw4YNceyxx8ZDDz20xSd0AgDkQ1GSJEm+i2gOmUwmKioqoq6uznMxAIBtpocofK4RALA9trWH8MwzAAAAAEghPAMAAACAFMIzAAAAAEghPAMAAACAFMIzAAAAAEghPAMAAACAFMIzAAAAAEghPAMAAACAFMIzAAAAAEghPAMAAACAFAUZns2aNSuKiopiwoQJ2bE333wzhg0bFh07dozy8vIYPnx4rFu3Ln9FAgAAALDLK7jwbNmyZXHDDTfEoYcemh379NNPY+DAgVFUVBSLFi2Kp556Kr788ssYMmRIbN68OY/VAgAAALArK6jwbOPGjTFy5Mi48cYbY4899siOP/XUU/H222/HLbfcEn369Ik+ffrE3LlzY/ny5bFo0aI8VgwAAADArqygwrPx48fH4MGDo6amptF4fX19FBUVRVlZWXasdevWUVxcHIsXL27uMgEAAABoIQomPJs/f34899xzUVtbu8W+Y445Jtq2bRuTJ0+Ozz77LD799NO4+OKLo6GhIdasWdPk8err6yOTyTTaAAAAAOCbKIjwbPXq1XHBBRfEvHnzonXr1lvs79ixY9x1113xwAMPxO677x4VFRWxYcOGOPLII6O4uOlTqK2tjYqKiuzWpUuXXJ8GAAAAALuYoiRJknwXcd9998WwYcOipKQkO9bQ0BBFRUVRXFwc9fX12X0ffvhhlJaWRvv27aOysjIuuuiimDRp0hbHrK+vj/r6+uzXmUwmunTpEnV1dVFeXp77kwIAdgmZTCYqKir0EAXMNQIAtse29hClzVhTqgEDBsTKlSsbjY0ePTp69eoVkydPbhSq7bXXXhERsWjRoli/fn2cfPLJTR6zrKys0TPSAAAAAOCbKojwrF27dnHIIYc0Gmvbtm106NAhO/6rX/0qevfuHR07doylS5fGBRdcEBMnToyDDjooHyUDAAAA0AIURHi2LVatWhVTp06Njz/+OLp37x6XXHJJTJw4Md9lAQAAALALK4hnnjUHz8IAALaHHqLwuUYAwPbY1h6iID5tEwAAAAAKkfAMAAAAAFIIzwAAAAAghfAMAAAAAFIIzwAAAAAghfAMAAAAAFIIzwAAAAAghfAMAAAAAFIIzwAAAAAghfAMAAAAAFIIzwAAAAAghfAMAAAAAFIIzwAAAAAghfAMAAAAAFIIzwAAAAAghfAMAAAAAFIIzwAAAAAghfAMAAAAAFIIzwAAAAAghfAMAAAAAFIIzwAAAAAghfAMAAAAAFIIzwAAAAAghfAMAAAAAFIIzwAAAAAghfAMAAAAAFIIzwAAAAAghfAMAAAAAFIIzwAAAAAghfAMAAAAAFIIzwAAAAAghfAMAAAAAFIIzwAAAAAghfAMAAAAAFIIzwAAAAAghfAMAAAAAFIIzwAAAAAghfAMAAAAAFIIzwAAAAAghfAMAAAAAFIIzwAAyJmGhoaYNm1a9OjRI9q0aRM9e/aMGTNmRJIk2TkbN26Mc889Nzp37hxt2rSJgw8+OK6//vo8Vg0A8Gel+S4AAIBd1xVXXBHXXXddzJ07N6qrq2P58uUxevToqKioiPPPPz8iIi688MJYtGhR3H777dG9e/f43e9+F+PGjYuqqqo4+eST83wGAEBL584zAAByZsmSJTF06NAYPHhwdO/ePU477bQYOHBgPPvss43mnHnmmfGd73wnunfvHmPGjInDDjus0RwAgHwRngEAkDP9+/ePhQsXxmuvvRYRES+++GIsXrw4Bg0a1GjO/fffH++//34kSRKPPvpovPbaazFw4MB8lQ0AkOVtmwAA5MyUKVMik8lEr169oqSkJBoaGmLmzJkxcuTI7Jw5c+bEmDFjonPnzlFaWhrFxcVx4403xnHHHdfkMevr66O+vj77dSaTyfl5AAAtl/AMAICcWbBgQcybNy/uuOOOqK6ujhdeeCEmTJgQVVVVceaZZ0bEn8Kzp59+Ou6///7o1q1bPPHEEzF+/PioqqqKmpqaLY5ZW1sb06dPb+5TAQBaqKLkLz/qaBeWyWSioqIi6urqory8PN/lAAA7CT3E36ZLly4xZcqUGD9+fHbspz/9adx+++3x6quvxueffx4VFRVx7733xuDBg7NzfvSjH8V7770XDz300BbHbOrOsy5durhGAMA3sq19njvPAADImc8++yyKixs/ZrekpCQ2b94cERGbNm2KTZs2bXXO15WVlUVZWVluCgYA+BrhGQAAOTNkyJCYOXNmdO3aNaqrq+P555+Pa665Js4666yIiCgvL4/jjz8+Jk2aFG3atIlu3brF448/Hrfeemtcc801ea4eAEB4BgBADs2ZMyemTZsW48aNi/Xr10dVVVWMHTs2Lrvssuyc+fPnx9SpU2PkyJHx8ccfR7du3WLmzJlxzjnn5LFyAIA/8cwzAICt0EMUPtcIANge29pDFKfuAQAAAIAWTngGAAAAACmEZwAAAACQQngGAAAAACkKMjybNWtWFBUVxYQJE7Jja9eujVGjRkVlZWW0bds2jjzyyLjnnnvyVyQAAAAAu7yCC8+WLVsWN9xwQxx66KGNxs8444xYtWpV3H///bFy5co45ZRTYvjw4fH888/nqVIAAAAAdnUFFZ5t3LgxRo4cGTfeeGPssccejfYtWbIkzjvvvDj66KNjv/32i0svvTTat28fK1asyFO1AAAAAOzqCio8Gz9+fAwePDhqamq22Ne/f/+488474+OPP47NmzfH/Pnz44svvojvfOc7TR6rvr4+MplMow0AAAAAvonSfBfwlfnz58dzzz0Xy5Yta3L/ggUL4gc/+EF06NAhSktL41vf+lbce++9sf/++zc5v7a2NqZPn57LkgEAAADYxRXEnWerV6+OCy64IObNmxetW7ducs60adNiw4YN8V//9V+xfPnyuPDCC2P48OGxcuXKJudPnTo16urqstvq1atzeQoAAAAA7IKKkiRJ8l3EfffdF8OGDYuSkpLsWENDQxQVFUVxcXGsWrUq9t9//3jppZeiuro6O6empib233//uP766//q98hkMlFRURF1dXVRXl6ek/MAAHY9eojC5xoBANtjW3uIgnjb5oABA7a4g2z06NHRq1evmDx5cnz22WcREVFc3PhGuZKSkti8eXOz1QkAAABAy1IQ4Vm7du3ikEMOaTTWtm3b6NChQxxyyCGxadOm2H///WPs2LFx9dVXR4cOHeK+++6LRx55JP7jP/4jT1UDAAAAsKsriGee/TWtWrWKBx98MDp27BhDhgyJQw89NG699daYO3dufO9738t3eQAAAADsogrizrOmPPbYY42+PuCAA+Kee+7JTzEAAAAAtEg7xZ1nAAAAAJAPwjMAAAAASCE8AwAAAIAUwjMAAAAASCE8AwAAAIAUwjMAAAAASCE8AwAAAIAUwjMAAAAASCE8AwAAAIAUwjMAAAAASCE8AwAAAIAUwjMAAAAASCE8AwAAAIAUwjMAAAAASCE8AwAAAIAUwjMAAAAASCE8AwAAAIAUwjMAAAAASCE8AwAAAIAUwjMAAAAASCE8AwAAAIAUwjMAAAAASCE8AwAAAIAUwjMAAAAASCE8AwAAAIAUwjMAAAAASCE8AwAAAIAUwjMAAAAASCE8AwAAAIAUwjMAAAAASCE8AwAAAIAUwjMAAAAASCE8AwAAAIAUwjMAAAAASCE8AwAAAIAUwjMAAAAASCE8AwAAAIAUwjMAAAAASCE8AwAAAIAUwjMAAAAASCE8AwAAAIAUwjMAAAAASCE8AwAAAIAUwjMAAAAASCE8AwAAAIAUwjMAAAAASCE8AwAgZxoaGmLatGnRo0ePaNOmTfTs2TNmzJgRSZI0mvfKK6/EySefHBUVFdG2bdv49re/He+++26eqgYA+LPSfBcAAMCu64orrojrrrsu5s6dG9XV1bF8+fIYPXp0VFRUxPnnnx8REW+++WYce+yxcfbZZ8f06dOjvLw8fv/730fr1q3zXD0AgPAMAIAcWrJkSQwdOjQGDx4cERHdu3ePX//61/Hss89m51xyySXxve99L6688srsWM+ePZu9VgCApnjbJgAAOdO/f/9YuHBhvPbaaxER8eKLL8bixYtj0KBBERGxefPm+O1vfxsHHnhgnHjiibH33ntH375947777stj1QAAfyY8AwAgZ6ZMmRKnn3569OrVK1q1ahVHHHFETJgwIUaOHBkREevXr4+NGzfGrFmz4qSTTorf/e53MWzYsDjllFPi8ccfb/KY9fX1kclkGm0AALnibZsAAOTMggULYt68eXHHHXdEdXV1vPDCCzFhwoSoqqqKM888MzZv3hwREUOHDo2JEydGRMThhx8eS5Ysieuvvz6OP/74LY5ZW1sb06dPb9bzAABaLneeAQCQM5MmTcrefdanT58YNWpUTJw4MWprayMiYq+99orS0tI4+OCDG72ud+/eqZ+2OXXq1Kirq8tuq1evzvl5AAAtlzvPAADImc8++yyKixv/vbakpCR7x9luu+0W3/72t2PVqlWN5rz22mvRrVu3Jo9ZVlYWZWVluSkYAOBrhGcAAOTMkCFDYubMmdG1a9eorq6O559/Pq655po466yzsnMmTZoUP/jBD+K4446LE044IR566KF44IEH4rHHHstf4QAA/68g37Y5a9asKCoqigkTJkRExNtvvx1FRUVNbnfddVd+iwUAINWcOXPitNNOi3HjxkXv3r3j4osvjrFjx8aMGTOyc4YNGxbXX399XHnlldGnT5/4t3/7t7jnnnvi2GOPzWPlAAB/UpQkSZLvIv7SsmXLYvjw4VFeXh4nnHBCzJ49OxoaGuJ///d/G83713/917jqqqtizZo1sfvuu//V42YymaioqIi6urooLy/PVfkAwC5GD1H4XCMAYHtsaw9RUHeebdy4MUaOHBk33nhj7LHHHtnxkpKSqKysbLTde++9MXz48G0KzgAAAABgexRUeDZ+/PgYPHhw1NTUbHXeihUr4oUXXoizzz47dU59fX1kMplGGwAAAAB8EwXzgQHz58+P5557LpYtW/ZX5950003Ru3fv6N+/f+qc2tramD59+o4sEQAAAIAWpiDuPFu9enVccMEFMW/evGjduvVW537++edxxx13bPWus4iIqVOnRl1dXXZbvXr1jiwZAAAAgBagIO48W7FiRaxfvz6OPPLI7FhDQ0M88cQT8Ytf/CLq6+ujpKQkIiLuvvvu+Oyzz+KMM87Y6jHLysqirKwsp3UDAAAAsGsriPBswIABsXLlykZjo0ePjl69esXkyZOzwVnEn96yefLJJ0fHjh2bu0wAAAAAWpiCCM/atWsXhxxySKOxtm3bRocOHRqNv/HGG/HEE0/Egw8+2NwlAgAAANACFcQzz7bVzTffHJ07d46BAwfmuxQAAAAAWoCiJEmSfBfRHDKZTFRUVERdXV2Ul5fnuxwAYCehhyh8rhEAsD22tYfYqe48AwAAAIDmJDwDAAAAgBTCMwAAAABIITwDAAAAgBTCMwAAAABIITwDAAAAgBTCMwAAAABIITwDAAAAgBTCMwAAAABIITwDAAAAgBTCMwAAAABIITwDAAAAgBTCMwAAAABIITwDAAAAgBTCMwAAAABIITwDAAAAgBTCMwAAAABIITwDAAAAgBTCMwAAAABIITwDAAAAgBTCMwAAAABIITwDAAAAgBTCMwAAAABIITwDAAAAgBTCMwAAAABIITwDAAAAgBTCMwAAAABIITwDAAAAgBTCMwAAAABIITwDAAAAgBTCMwAAAABIITwDAAAAgBTCMwAAAABIITwDAAAAgBTCMwAAAABIITwDAAAAgBTCMwAAAABIITwDAAAAgBTCMwAAAABIITwDAAAAgBTCMwAAAABIITwDAAAAgBTCMwAAAABIITwDAAAAgBTCMwAAAABIITwDAAAAgBTCMwAAAABIITwDAAAAgBTCMwAAAABIITwDAAAAgBTCMwAAAABIITwDAAAAgBTCMwAAcqahoSGmTZsWPXr0iDZt2kTPnj1jxowZkSRJk/PPOeecKCoqitmzZzdvoQAAKUrzXQAAALuuK664Iq677rqYO3duVFdXx/Lly2P06NFRUVER559/fqO59957bzz99NNRVVWVp2oBALZUkHeezZo1K4qKimLChAmNxpcuXRrf/e53o23btlFeXh7HHXdcfP755/kpEgCAv2rJkiUxdOjQGDx4cHTv3j1OO+20GDhwYDz77LON5r3//vtx3nnnxbx586JVq1Z5qhYAYEsFF54tW7Ysbrjhhjj00EMbjS9dujROOumkbLO1bNmyOPfcc6O4uOBOAQCA/9e/f/9YuHBhvPbaaxER8eKLL8bixYtj0KBB2TmbN2+OUaNGxaRJk6K6ujpfpQIANKmg3ra5cePGGDlyZNx4443x05/+tNG+iRMnxvnnnx9TpkzJjh100EHNXSIAAN/AlClTIpPJRK9evaKkpCQaGhpi5syZMXLkyOycK664IkpLS7d4G2ea+vr6qK+vz36dyWR2eN0AAF8pqNu2xo8fH4MHD46amppG4+vXr49nnnkm9t577+jfv3906tQpjj/++Fi8eHGeKgUAYFssWLAg5s2bF3fccUc899xzMXfu3Lj66qtj7ty5ERGxYsWKuPbaa+OWW26JoqKibTpmbW1tVFRUZLcuXbrk8hQAgBauYMKz+fPnx3PPPRe1tbVb7Puf//mfiIj4yU9+Ej/+8Y/joYceiiOPPDIGDBgQr7/+epPHq6+vj0wm02gDAKB5TZo0KaZMmRKnn3569OnTJ0aNGhUTJ07M9nxPPvlkrF+/Prp27RqlpaVRWloa77zzTlx00UXRvXv3Jo85derUqKury26rV69uxjMCAFqagnjb5urVq+OCCy6IRx55JFq3br3F/s2bN0dExNixY2P06NEREXHEEUfEwoUL4+abb24ycKutrY3p06fntnAAALbqs88+2+IZtSUlJdn+btSoUVu86+DEE0+MUaNGZfu+rysrK4uysrLcFAwA8DUFEZ6tWLEi1q9fH0ceeWR2rKGhIZ544on4xS9+EatWrYqIiIMPPrjR63r37h3vvvtuk8ecOnVqXHjhhdmvM5mMW/oBAJrZkCFDYubMmdG1a9eorq6O559/Pq655po466yzIiKiQ4cO0aFDh0avadWqVVRWVnq+LQBQEAoiPBswYECsXLmy0djo0aOjV69eMXny5Nhvv/2iqqoqG6J95bXXXmv0SU1/yV8kAQDyb86cOTFt2rQYN25crF+/PqqqqmLs2LFx2WWX5bs0AIBtUhDhWbt27eKQQw5pNNa2bdvo0KFDdnzSpElx+eWXx2GHHRaHH354zJ07N1599dW4++6781EyAADboF27djF79uyYPXv2Nr/m7bffzlk9AADfVEGEZ9tiwoQJ8cUXX8TEiRPj448/jsMOOyweeeSR6NmzZ75LAwAAAGAXVZQkSZLvIppDJpOJioqKqKuri/Ly8nyXAwDsJPQQhc81AgC2x7b2EMWpewAAAACghROeAQAAAEAK4RkAAAAApBCeAQAAAEAK4RkAAAAApBCeAQAAAEAK4RkAAAAApBCeAQAAAEAK4RkAAAAApBCeAQAAAEAK4RkAAAAApCjNdwHNJUmSiIjIZDJ5rgQA2Jl81Tt81UtQePR5AMD22NY+r8WEZ5988klERHTp0iXPlQAAO6NPPvkkKioq8l0GTdDnAQB/i7/W5xUlLeTPqJs3b44PPvgg2rVrF0VFRfkup+BkMpno0qVLrF69OsrLy/NdTotj/fPPNcgv659f1n/rkiSJTz75JKqqqqK42BMvCpE+b+v8juefa5Bf1j+/rH9+Wf+t29Y+r8XceVZcXBydO3fOdxkFr7y83C9UHln//HMN8sv655f1T+eOs8Kmz9s2fsfzzzXIL+ufX9Y/v6x/um3p8/z5FAAAAABSCM8AAAAAIIXwjIiIKCsri8svvzzKysryXUqLZP3zzzXIL+ufX9Yfdm1+x/PPNcgv659f1j+/rP+O0WI+MAAAAAAAvil3ngEAAABACuEZAAAAAKQQngEAAABACuEZAAAAAKQQnrUgH3/8cYwcOTLKy8ujffv2cfbZZ8fGjRu3+povvvgixo8fHx06dIjdd989Tj311Fi3bl2Tcz/66KPo3LlzFBUVxYYNG3JwBju3XKz/iy++GCNGjIguXbpEmzZtonfv3nHttdfm+lR2Cv/yL/8S3bt3j9atW0ffvn3j2Wef3er8u+66K3r16hWtW7eOPn36xIMPPthof5Ikcdlll8U+++wTbdq0iZqamnj99ddzeQo7tR25/ps2bYrJkydHnz59om3btlFVVRVnnHFGfPDBB7k+jZ3Wjv75/0vnnHNOFBUVxezZs3dw1cDfQp+XX/q85qXPyy99Xn7p8/IkocU46aSTksMOOyx5+umnkyeffDLZf//9kxEjRmz1Neecc07SpUuXZOHChcny5cuTY445Junfv3+Tc4cOHZoMGjQoiYjkD3/4Qw7OYOeWi/W/6aabkvPPPz957LHHkjfffDO57bbbkjZt2iRz5szJ9ekUtPnz5ye77bZbcvPNNye///3vkx//+MdJ+/btk3Xr1jU5/6mnnkpKSkqSK6+8Mnn55ZeTSy+9NGnVqlWycuXK7JxZs2YlFRUVyX333Ze8+OKLycknn5z06NEj+fzzz5vrtHYaO3r9N2zYkNTU1CR33nln8uqrryZLly5Njj766OSoo45qztPaaeTi5/8r//7v/54cdthhSVVVVfKzn/0sx2cCfBP6vPzS5zUffV5+6fPyS5+XP8KzFuLll19OIiJZtmxZduw///M/k6KiouT9999v8jUbNmxIWrVqldx1113ZsVdeeSWJiGTp0qWN5v7yl79Mjj/++GThwoWaqibkev3/0rhx45ITTjhhxxW/Ezr66KOT8ePHZ79uaGhIqqqqktra2ibnDx8+PBk8eHCjsb59+yZjx45NkiRJNm/enFRWViZXXXVVdv+GDRuSsrKy5Ne//nUOzmDntqPXvynPPvtsEhHJO++8s2OK3oXkav3fe++9ZN99901eeumlpFu3bpoqKCD6vPzS5zUvfV5+6fPyS5+XP9622UIsXbo02rdvH3/3d3+XHaupqYni4uJ45plnmnzNihUrYtOmTVFTU5Md69WrV3Tt2jWWLl2aHXv55Zfjn//5n+PWW2+N4mI/Uk3J5fp/XV1dXey55547rvidzJdffhkrVqxotG7FxcVRU1OTum5Lly5tND8i4sQTT8zOf+utt2Lt2rWN5lRUVETfvn23ei1aolysf1Pq6uqiqKgo2rdvv0Pq3lXkav03b94co0aNikmTJkV1dXVuige2mz4vv/R5zUefl1/6vPzS5+WX/wO2EGvXro2999670VhpaWnsueeesXbt2tTX7Lbbblv8R6tTp07Z19TX18eIESPiqquuiq5du+ak9l1Brtb/65YsWRJ33nlnjBkzZofUvTP68MMPo6GhITp16tRofGvrtnbt2q3O/+qf3+SYLVUu1v/rvvjii5g8eXKMGDEiysvLd0zhu4hcrf8VV1wRpaWlcf755+/4ooG/mT4vv/R5zUefl1/6vPzS5+WX8GwnN2XKlCgqKtrq9uqrr+bs+0+dOjV69+4dP/zhD3P2PQpZvtf/L7300ksxdOjQuPzyy2PgwIHN8j2huW3atCmGDx8eSZLEddddl+9yWoQVK1bEtddeG7fccksUFRXluxxoUfLdZ+jz9HnQnPR5zU+ft+1K810Af5uLLroo/vEf/3Grc/bbb7+orKyM9evXNxr/4x//GB9//HFUVlY2+brKysr48ssvY8OGDY3+KrZu3brsaxYtWhQrV66Mu+++OyL+9Ek1ERF77bVXXHLJJTF9+vTtPLOdQ77X/ysvv/xyDBgwIMaMGROXXnrpdp3LrmKvvfaKkpKSLT4trKl1+0plZeVW53/1z3Xr1sU+++zTaM7hhx++A6vf+eVi/b/yVUP1zjvvxKJFi/w1sgm5WP8nn3wy1q9f3+iuk4aGhrjoooti9uzZ8fbbb+/YkwCy8t1n6PP0eYVGn5df+rz80uflWX4fuUZz+epBpsuXL8+OPfzww9v0INO77747O/bqq682epDpG2+8kaxcuTK73XzzzUlEJEuWLEn9xI+WKFfrnyRJ8tJLLyV77713MmnSpNydwE7m6KOPTs4999zs1w0NDcm+++671Qdp/sM//EOjsX79+m3xINmrr746u7+urs6DZFPs6PVPkiT58ssvk+9///tJdXV1sn79+twUvovY0ev/4YcfNvrv/MqVK5Oqqqpk8uTJyauvvpq7EwG2mT4vv/R5zUufl1/6vPzS5+WP8KwFOemkk5IjjjgieeaZZ5LFixcnBxxwQKOP0H7vvfeSgw46KHnmmWeyY+ecc07StWvXZNGiRcny5cuTfv36Jf369Uv9Ho8++qhPYUqRi/VfuXJl0rFjx+SHP/xhsmbNmuzW0v+nM3/+/KSsrCy55ZZbkpdffjkZM2ZM0r59+2Tt2rVJkiTJqFGjkilTpmTnP/XUU0lpaWly9dVXJ6+88kpy+eWXN/kR5u3bt09+85vfJP/93/+dDB061EeYp9jR6//ll18mJ598ctK5c+fkhRdeaPSzXl9fn5dzLGS5+Pn/Op/CBIVHn5df+rzmo8/LL31efunz8kd41oJ89NFHyYgRI5Ldd989KS8vT0aPHp188skn2f1vvfVWEhHJo48+mh37/PPPk3HjxiV77LFH8q1vfSsZNmxYsmbNmtTvoalKl4v1v/zyy5OI2GLr1q1bM55ZYZozZ07StWvXZLfddkuOPvro5Omnn87uO/7445Mzzzyz0fwFCxYkBx54YLLbbrsl1dXVyW9/+9tG+zdv3pxMmzYt6dSpU1JWVpYMGDAgWbVqVXOcyk5pR67/V78bTW1/+fvCn+3on/+v01RB4dHn5Zc+r3np8/JLn5df+rz8KEqS/394AQAAAADQiE/bBAAAAIAUwjMAAAAASCE8AwAAAIAUwjMAAAAASCE8AwAAAIAUwjMAAAAASCE8AwAAAIAUwjMAAAAASCE8AwAAAIAUwjMAAAAASCE8AwAAAIAUwjMAAAAASPF/9dvU/+1tdjcAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# !pip install torchsummary\n",
        "# from torchsummary import summary\n",
        "# use_cuda = torch.cuda.is_available()\n",
        "# device = torch.device(\"cuda\" if use_cuda else \"cpu\")\n",
        "# model = Net().to(device)\n",
        "# summary(model, input_size=(1, 28, 28))\n",
        "\n",
        "from utils import model_summary\n",
        "model_summary(Net)"
      ],
      "metadata": {
        "id": "C8WZPfXe4iK_",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "452fd378-d9c9-4c5d-c42b-f717f29c7d05"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "----------------------------------------------------------------\n",
            "        Layer (type)               Output Shape         Param #\n",
            "================================================================\n",
            "            Conv2d-1           [-1, 32, 26, 26]             288\n",
            "            Conv2d-2           [-1, 64, 24, 24]          18,432\n",
            "            Conv2d-3          [-1, 128, 10, 10]          73,728\n",
            "            Conv2d-4            [-1, 256, 8, 8]         294,912\n",
            "            Linear-5                   [-1, 50]         204,800\n",
            "            Linear-6                   [-1, 10]             500\n",
            "================================================================\n",
            "Total params: 592,660\n",
            "Trainable params: 592,660\n",
            "Non-trainable params: 0\n",
            "----------------------------------------------------------------\n",
            "Input size (MB): 0.00\n",
            "Forward/backward pass size (MB): 0.67\n",
            "Params size (MB): 2.26\n",
            "Estimated Total Size (MB): 2.93\n",
            "----------------------------------------------------------------\n"
          ]
        }
      ]
    }
  ]
}
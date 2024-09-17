import torch
import torch.nn as nn

input_img = torch.rand(1, 3, 7, 7)
layer = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=2, padding=1)
out = layer(input_img)
assert out.shape == torch.Size([1, 6, 4, 4])


input_img = torch.rand(1, 3, 8, 8)
layer = nn.MaxPool2d(kernel_size=2, stride=2)
out = layer(input_img)
assert out.shape == torch.Size([1, 3, 4, 4])


import os
import sys

cwd = os.getcwd()
# add CIFAR10 data in the environment
sys.path.append(cwd + "/../cifar10")

# Numpy is linear algebra lbrary
import numpy as np

# Matplotlib is a visualizations library
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import utils
from torchvision import transforms

# CIFAR10 is a custom Dataloader that loads a subset ofthe data from a local folder
from Cifar10Dataloader import CIFAR10

batch_size = 4


def show_image(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def load_data():

    # convert the images to tensor and normalized them
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = CIFAR10(root="../cifar10", transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False, num_workers=1
    )
    return trainloader


## Define a CNN
# Here you will come into play. Try to define the necessary layers and build the forward pass of our model. Remember that the model's structure is:


# - A conv layer with 3 channels as input, 6 channels as output, and a 5x5 kernel
# - A 2x2 max-pooling layer
# - A conv layer with 6 channels as input, 16 channels as output, and a 5x5 kernel
# - A linear layer with 1655 nodes
# - A linear layer with 120 nodes
# - A linear layer with 84 nodes
# - A linear layer with 10 nodes


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = CNN()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 2. TRAIN THE MODEL
def train(model, training_data):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    running_loss = 0.0

    for epoch in range(1):  # loop over the dataset multiple times

        for i, data in enumerate(training_data, 0):
            # get the inputs; cifar10 is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print("Finished Training")


# -----------------------------------------------------------------------------------------------------------------------

# Let’s now implement batch normalization from scratch for images of size [N, C, H, W]. All you have to do is transform the above equations to Pytorch. The tricky part is to correctly figure out the sizes of each tensor.
def batchnorm(X, gamma, beta):
    """
    Batch norm in Pytorch
    """
    # Mean and standard deviation of the batch
    mu = torch.mean(X, dim=0)
    sigma = torch.std(X, dim=0)

    # Normalize the batch
    X_norm = (X - mu) / sigma

    # Scale and shift
    X_norm = X_norm * gamma + beta

    return X_norm


# -----------------------------------------------------------------------------------------------------------------------
# Dropout

inp = torch.rand(1, 8)
layer = nn.Dropout(0.5)
out1 = layer(inp)
out2 = layer(inp)
assert out1.shape == torch.Size([1, 8])
assert out2.shape == torch.Size([1, 8])

# -----------------------------------------------------------------------------------------------------------------------
# Skip Connections
# Your goal is to write the forward function so that the input is added to the output of the two layers, forming a residual connection. In essence, you will represent the exact above image in code.

seed = 172
torch.manual_seed(seed)


class SkipConnection(nn.Module):
    def __init__(self):
        super(SkipConnection, self).__init__()
        self.conv_layer1 = nn.Conv2d(3, 6, 2, stride=2, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv_layer2 = nn.Conv2d(6, 3, 2, stride=2, padding=2)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.conv_layer1(x)
        x = self.relu(x)
        # use skip connection
        x = x + self.conv_layer2(x)
        x = self.relu2(x)
        return x


# -----------------------------------------------------------------------------------------------------------------------
# AlexNet

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


model = AlexNet(num_classes=10)
inp = torch.rand(1, 3, 128, 128)
print(model(inp).shape)

# -----------------------------------------------------------------------------------------------------------------------
# Inception/GoogLeNet

import torch
import torch.nn as nn


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        relu = nn.ReLU()
        self.branch1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            relu,
        )

        conv3_1 = nn.Conv2d(
            in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0
        )
        conv3_3 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.branch2 = nn.Sequential(conv3_1, conv3_3, relu)

        conv5_1 = nn.Conv2d(
            in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0
        )
        conv5_5 = nn.Conv2d(
            out_channels, out_channels, kernel_size=5, stride=1, padding=2
        )
        self.branch3 = nn.Sequential(conv5_1, conv5_5, relu)

        max_pool_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv_max_1 = nn.Conv2d(
            in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0
        )
        self.branch4 = nn.Sequential(max_pool_1, conv_max_1, relu)

    def forward(self, input):
        output1 = self.branch1(input)
        output2 = self.branch2(input)
        output3 = self.branch3(input)
        output4 = self.branch4(input)
        return torch.cat([output1, output2, output3, output4], dim=1)


model = InceptionModule(in_channels=3, out_channels=32)
inp = torch.rand(1, 3, 128, 128)
print(model(inp).shape)

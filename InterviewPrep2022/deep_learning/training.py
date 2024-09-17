# Adam - momentum and adaptive learning rate
# for t in range(steps):
#     dw = gradient(loss, w)
#     moment1= delta1 *moment1  +(1-delta1)* dw
#     moment2 = delta2*moment2 +(1-delta2)*dw*dw
#     moment1_unbiased = moment1  /(1-delta1**t)
#     moment2_unbiased = moment2  /(1-delta2**t)
#     w = w - learning_rate*moment1_unbiased/ (moment2_unbiased.sqrt()+e)

import torch

seed = 172
torch.manual_seed(seed)


def m_sigmoid(x):
    return 1 / (1 + torch.exp(-x))
    # return 1 / (1+math.exp(-x))


def m_tanh(x):
    return torch.tanh(x)


def m_relu(x):
    return x


# return torch.max(torch.tensor([0],dtype=torch.LongTensor), x)


def m_softmax(x):
    return torch.exp(x) / sum(torch.exp(x))


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
import torch.optim as optim
from torchvision import utils
from torchvision import transforms

# CIFAR10 is a custom Dataloader that loads a subset ofthe data from a local folder
from Cifar10Dataloader import CIFAR10


batch_size = 4


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

# get some random training images
dataiter = iter(load_data())
images, labels = dataiter.next()

# show images
show_image(utils.make_grid(images))
# print labels
print(" ".join("%5s" % classes[labels[j]] for j in range(4)))


## 1. DEFINE MODEL HERE
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(3072, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu2(x)
        x = self.l3(x)
        return x


def train():

    model = Model()
    training_data = load_data()

    # 2. LOSS AND OPTIMIZER
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    running_loss = 0.0

    for epoch in range(10):
        for i, data in enumerate(training_data, 0):

            inputs, labels = data
            # reshape images so they can be fed to a nn.Linear()
            inputs = inputs.view(inputs.size(0), -1)

            optimizer.zero_grad()

            ##3. RUN BACKPROP
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:  # print every 500 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

    print("Training finished")


train()


def evaluate():
    dataiter = iter(load_data())
    images, labels = dataiter.next()

    # print images
    show_image(utils.make_grid(images))
    print("GroundTruth: ", " ".join("%5s" % classes[labels[j]] for j in range(4)))

    images = images.view(images.size(0), -1)
    outputs = model(images)

    _, predicted = torch.max(outputs, 1)

    print("Predicted: ", " ".join("%5s" % classes[predicted[j]] for j in range(4)))


evaluate()

## 1. DEFINE MODEL
model = nn.Sequential(
    nn.Linear(3072, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10)
)

##2. LOSS AND OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

##3. RUN BACKPROP
outputs = model(inputs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()

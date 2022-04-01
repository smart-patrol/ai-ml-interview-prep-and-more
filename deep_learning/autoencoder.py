import torch
import torch.nn as nn

seed = 172
torch.manual_seed(seed)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [-1, 12, 16, 16]
        self.conv2d_1 = nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        # ouput size: [-1, 24, 8, 8]
        self.conv2d_2 = nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        # ouput size: [-1, 48, 4, 4]
        self.conv2d_3 = nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        # ouput size: [-1, 96, 2, 2]
        self.conv2d_4 = nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        # ouput size: [-1, 48, 4, 4]
        self.conv_transpose2d_1 = nn.ConvTranspose2d(96, 48, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        # ouput size: [-1, 24, 8, 8]
        self.conv_transpose2d_2 = nn.ConvTranspose2d(48, 24, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU()
        # ouput size: [-1, 12, 16, 16]
        self.conv_transpose2d_3 = nn.ConvTranspose2d(24, 12, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.ReLU()
        # ouput size: [-1, 3, 32, 32]
        self.conv_transpose2d_4 = nn.ConvTranspose2d(12, 3, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.conv2d_2(x)
        x = self.relu2(x)
        x = self.conv2d_3(x)
        x = self.relu3(x)
        x = self.conv2d_4(x)
        x = self.relu4(x)
        x = self.conv_transpose2d_1(x)
        x = self.relu5(x)
        x = self.conv_transpose2d_2(x)
        x = self.relu6(x)
        x = self.conv_transpose2d_3(x)
        x = self.relu7(x)
        x = self.conv_transpose2d_4(x)
        x = self.sigmoid(x)
        return x


#-----------------------------------------------------------------------------------------------------------------------
def elbo(reconstructed, input, mu, logvar):
     """
        Args:
            `reconstructed`: The reconstructed input of size [B, C, W, H],
            `input`: The oriinal input of size [B, C, W, H],
            `mu`: The mean of the Gaussian of size [N], where N is the latent dimension
            `logvar`: The log of the variance of the Gaussian of size [N], where N is the latent dimension

        Returns:
            a scalar
     """
     # Reconstruction loss
     BCE = nn.functional.binary_cross_entropy(reconstructed, input, reduction='sum')
     # KL divergence
     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
     return BCE + KLD

def reparameterize(mu, log_var):
    """
        Args:
            `mu`: mean from the encoder's latent space
            `log_var`: log variance from the encoder's latent space

        Returns:
            the reparameterized latent vector z
    """
    std = torch.exp(0.5 * log_var)  # standard deviation
    eps = torch.randn_like(std)  # generate sample of the same size
    sample = mu + (eps * std)  # sampling as if coming from the input space
    return sample


#-----------------------------------------------------------------------------------------------------------------------

import os
import sys
cwd = os.getcwd()
#add CIFAR10 data in the environment
sys.path.append(cwd + '/../cifar10') 
from Cifar10Dataloader import CIFAR10
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import utils
from torchvision import transforms

seed = 172
torch.manual_seed(seed)

batch_size=4

def load_data():
    
    #convert the images to tensor and normalized them
    transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    trainset = CIFAR10(root='../cifar10',  transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, num_workers=1)
    return trainloader


def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.features =16
        # encoder
        self.enc1 = nn.Linear(in_features=3072, out_features=128)
        self.enc2 = nn.Linear(in_features=128, out_features=self.features * 2)

        # decoder
        self.dec1 = nn.Linear(in_features=self.features, out_features=128)
        self.dec2 = nn.Linear(in_features=128, out_features=3072)

    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = self.enc2(x).view(-1, 2, self.features)
        # get `mu` and `log_var`
        mu = x[:, 0, :]  # the first feature values as mean
        log_var = x[:, 1, :]  # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)

        # decoding
        x = F.relu(self.dec1(z))
        reconstruction = torch.sigmoid(self.dec2(x))
        return reconstruction, mu, log_var
        
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # generate sample of the same size
        sample = mu + (eps * std)  # sampling as if coming from the input space
        return sample

def train(model,training_data):
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss(reduction='sum')

    running_loss = 0.0

    for epoch in range(1):  # loop over the dataset multiple times

        for i, data in enumerate(training_data, 0):
            inputs, _ = data
            inputs = inputs.view(inputs.size(0), -1)

            optimizer.zero_grad()
            reconstruction, mu, logvar = model(inputs)
            bce_loss = criterion(reconstruction, inputs)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    PATH = './cifar_net.pth'
    torch.save(model.state_dict(), PATH)

    print('Finished Training')


model = VAE()
training_data = load_data()

train(model, training_data)
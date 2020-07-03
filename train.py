
import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(784,256)
        self.relu = nn.ReLU(True)
        self.lin2 = nn.Linear(256,64)
        self.lin3 = nn.Linear(64,256)
        self.lin4 = nn.Linear(256,784)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.relu(self.lin3(x))
        x = self.sig(self.lin4(x))
        return x

def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor


def addNoise(img):
    """Adds Gaussian Noise to input image"""
    return img + torch.randn_like(img)*4e-1


if __name__ == "__main__":
    # Training hyper parameters
    epochs = 200
    bs = 128
    lr = 1e-3
    model = Autoencoder().cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    dataset = MNIST('./data', transform=transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda tensor:min_max_normalization(tensor, 0, 1)),transforms.Lambda(lambda x:torch.round(x))]), download=True)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

    for epoch in range(epochs):
        running_loss = 0.
        for data in dataloader:
            optimizer.zero_grad()
            y = data[0].reshape(data[0].size(0), -1)
            x = addNoise(y)
            y_hat = model.forward(x)
            loss = criterion(y_hat, y)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        save_image(x.reshape(-1,1,28,28), f"image/x_{epoch}.png")
        save_image(y_hat.reshape(-1,1,28,28), f"image/y_hat_{epoch}.png")
        print(f'Epoch:{epoch+1}/{epochs}, Loss: {running_loss/len(dataloader)}')

    for i, data in enumerate(dataloader):
        y = data[0].reshape(data[0].size(0), -1)
        x = addNoise(y)
        y_hat = model.forward(x)
        y_hat = y_hat.reshape(-1,1,28,28)
        #save_image(data, f"./image/y_{i+1}.png")
        save_image(y_hat, f"image/y_hat_{i+1}.png")
        break

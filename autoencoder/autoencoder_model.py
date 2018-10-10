from torch import nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128),
                                     nn.ReLU(True),
                                     nn.Linear(128, 64),
                                     nn.ReLU(True),
                                     nn.Linear(64, 12))
        self.decoder = nn.Sequential(nn.Linear(12, 64), nn.ReLU(True),
                                     nn.Linear(64, 128),
                                     nn.ReLU(True),
                                     nn.Linear(128, 28*28),
                                     nn.Tanh())

    def forward(self, x):
        encoded_y = self.encoder(x)
        y = self.decoder(encoded_y)
        return y


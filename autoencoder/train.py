from torch import nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from autoencoder_model import Autoencoder

def main():
    num_epochs = 100
    batch_size = 128
    learning_rate = 1e-3

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = MNIST('./data', transform=img_transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    avg_losses = []
    num_batches = len(dataloader)
    for epoch in range(num_epochs):
        cumulative_loss = 0
        for batch in dataloader:
            imgs, _ = batch
            imgs = imgs.view(imgs.size(0), -1)
            imgs = Variable(imgs)
            output = model(imgs)
            loss = criterion(output, imgs)
            loss.backward()
            cumulative_loss += loss
            optimizer.step()
        avg_loss = cumulative_loss/num_batches
        avg_losses.append(avg_loss)
        if epoch % 10 == 0:
            print('loss is {}'.format(avg_loss))

if __name__ == '__main__':
    main()
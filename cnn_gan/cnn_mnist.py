import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.adam import Adam
from torchvision import datasets, transforms
from typing import List
from torch.utils.data import DataLoader, Subset
import tqdm
from copy import deepcopy
from argparse import ArgumentParser

from models import MAML, Network
from utils import powerset_without_null, display_images_with_prediction


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.c1 = nn.Conv2d(1, 16, 3, 2)  # 28x28->13x13
        self.c2 = nn.Conv2d(16, 32, 1, 1)
        self.c3 = nn.Conv2d(32, 64, 3, 2)  # 13-6
        self.c4 = nn.Conv2d(64, 64, 3, 2)  # 6-2
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.c1(x)))
        x = self.dropout(self.relu(self.c2(x)))
        x = self.dropout(self.relu(self.c3(x)))
        x = self.dropout(self.relu(self.c4(x)))
        x = x.view(-1, 64 * 2 * 2)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

    def predict(self, x):
        x = self.forward(x)
        return torch.multinomial(x / torch.sum(x), 1)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

    trainset = datasets.MNIST(
        "data",
        download=True,
        train=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    validset = datasets.MNIST(
        "data",
        download=True,
        train=False,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    batch_size = 4096
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=True)

    net = Classifier().to(device)
    optim = Adam(net.parameters(), 1e-3)

    one_hot_mat = F.one_hot(torch.arange(10)).float().to(device)

    print(
        f"training with {sum(p.numel() for p in net.parameters() if p.requires_grad)} parameters"
    )

    tbar = tqdm.trange(1000)
    for i in tbar:
        net_loss = 0.0
        tibar = tqdm.trange(len(train_loader), leave=False)
        shown = False
        for images, labels_hat in train_loader:
            images = images.to(device)
            labels = one_hot_mat[labels_hat]
            optim.zero_grad()
            pred = net(images)
            loss = F.cross_entropy(pred, labels)
            loss.backward()
            optim.step()
            net_loss += loss.item()
            tibar.update(1)
            if i % 50 == 0 and not shown:
                idx = np.random.randint(0, batch_size, 16)
                display_images_with_prediction(images[idx], labels_hat[idx], net)
                shown = True
        tibar.close()
        writer.add_scalar("classifier/net_loss", net_loss, i)
        # test validation set
        n_wrong = 0.0
        for images, labels_hat in valid_loader:
            images = images.to(device)
            pred = net.predict(images).flatten().cpu()
            labels = labels_hat.flatten()
            n_wrong += torch.sum(pred != labels)
        rate = 1.0 - n_wrong / (len(valid_loader) * batch_size)
        tbar.set_description(f"{net_loss:3.5f} {rate:3.5f}")
        writer.add_scalar("classifier/rate", rate, i)
    tbar.close()

    return


if __name__ == "__main__":
    parser = ArgumentParser()
    main(parser.parse_args())

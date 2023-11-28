import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx as onnx
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break
        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        return (
            img0,
            img1,
            torch.from_numpy(
                np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32)
            ),
        )

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print(f"Model loaded from {filepath}")
    return model


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + (label)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive


def train_siamese_network(model, train_loader, criterion, optimizer, epochs):
    counter = []
    loss_history = []
    iteration_number = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, (img0, img1, label) in enumerate(train_loader, 0):
            optimizer.zero_grad()
            output1, output2 = model(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            epoch_loss += loss_contrastive.item()
        epoch_loss /= len(train_loader)
        print(f"Epoch number {epoch}\n Average loss {epoch_loss}\n")
        iteration_number += 1
        counter.append(iteration_number)
        loss_history.append(epoch_loss)
    return counter, loss_history


def to_onnx(src, dst):
    model = load_model(SiameseNetwork(), src)
    input_shape = (1, 1, 100, 100)
    image1 = torch.randn(input_shape)
    image2 = torch.randn(input_shape)
    onnx.export(model, (image1, image2), dst, verbose=False)


def train(src, dst, epochs=200, lr=0.005, b=64):
    folder_dataset = datasets.ImageFolder(root=src)
    transformation = transforms.Compose(
        [transforms.Resize((100, 100)), transforms.ToTensor()]
    )
    siamese_dataset = SiameseNetworkDataset(
        imageFolderDataset=folder_dataset, transform=transformation
    )
    train_dataloader = DataLoader(
        siamese_dataset, shuffle=True, num_workers=8, batch_size=b
    )
    model = SiameseNetwork()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    counter, loss_history = train_siamese_network(
        model, train_dataloader, criterion, optimizer, epochs=epochs
    )
    plt.plot(counter, loss_history)
    plt.show()
    save_model(model, f"{dst}.pth")
    input_shape = (1, 1, 100, 100)
    image1 = torch.randn(input_shape)
    image2 = torch.randn(input_shape)
    onnx.export(model, (image1, image2), f"{dst}.onnx", verbose=False)


if __name__ == "__main__":
    cls = "pairs"
    epochs = 50
    lr = 0.00001
    b = 128
    train(
        src=f"./DATA/{cls}/training",
        dst=f"./MODELS/{cls}_e{epochs}_lr{lr}_b{b}",
        epochs=epochs,
        lr=lr,
        b=b,
    )
    print("##DONE")

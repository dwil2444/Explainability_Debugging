#!/bin/venv/python
import torchvision.models as models
import torch as torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from utils.helper import CleanCuda, GetDevice


WEIGHTPATH = 'weights/resnet50.pth'
resnet50 = models.resnet50(pretrained=False)


class CifarData():
    """
    """
    def __init__(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Resize([32, 32])
        ])
        self.val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Resize([32, 32])
        ])
        
    def get_dataset(self):
        """
        Uses torchvision.datasets.ImageNet to load dataset.
        Downloads dataset if doesn't exist already.
        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        trainset = datasets.CIFAR100('datasets/CIFAR100/train/', train='True', transform=self.train_transforms,
                                     target_transform=None, download=True)
        valset = datasets.CIFAR100('datasets/CIFAR100/val/', train='False', transform=self.val_transforms,
                                   target_transform=None, download=True)

        return trainset, valset 

    
    def filter_dataset(self, num_labels):
        """
        param: num_labels: the number of labels to
        remove from the dataset
        """
    
    
    def get_data_loader(self, batch_size=16):
        """
        Uses Class Object methods to generate
        torch dataloaders for train and val set
        
        param: batch_size: duh
        """
        trainset, valset = self.get_dataset()
        trainloader = DataLoader(trainset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2,
                            )
        valloader = DataLoader(valset,
                               batch_size=1,
                               shuffle=True,)
        return trainloader, valloader


def train_model(model, optimizer, dataloader, num_epochs=1, 
         criterion=nn.CrossEntropyLoss(), 
         ):
    """
    param: model: the image classifier

    param: optimizer: the optimization algorithm

    param: dataloader: torch.utils.dataloader with training data

    param: num_epochs: the number of epochs to train for

    param: criterion: the loss function
    """
    device = GetDevice()
    model.train()
    model = model.to(device)
    running_loss = 0
    for epoch in range(0, num_epochs):
        total_loss = 0
        for i, batch in enumerate(tqdm(dataloader)):
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch == 0:
            running_loss = total_loss
        else:
            if(running_loss > total_loss):
                running_loss = total_loss
                torch.save(model.state_dict(), WEIGHTPATH)
        print(f"Training loss: {total_loss} in Epoch: {epoch+1}")
    return model


def evaluate(model, dataloader):
    """
    param: model: the image classifier

    param: dataloader: torch.utils.dataloader with validation
    data
    """
    device = GetDevice()
    num_correct = 0
    num_seen = 0
    model.load_state_dict(torch.load(WEIGHTPATH))
    model.eval()
    sm = nn.Softmax(dim = 1)
    for i, batch in enumerate(tqdm(dataloader)):
            imgs,  label = batch
            imgs = imgs.to(device)
            label = label.to(device)
            outputs = model(imgs)
            pred = sm(outputs)
            pred = torch.argmax(pred, dim=1).item()
            num_seen += len(label)
            if(pred == label.item()):
                num_correct += 1
            else:
                pass
    acc = (num_correct/num_seen) * 100
    print(f"Validation Accuracy: {acc:.2f} %")
    return acc

def main():
    CleanCuda()
    trainloader, valloader = CifarData().get_data_loader(128)
    optimizer = optim.Adam(resnet50.parameters(),
                        lr=0.0001, eps=1e-08,)
    criterion = nn.CrossEntropyLoss()
    _ = train_model(resnet50, optimizer, trainloader, 20, criterion)
    evaluate(resnet50, valloader)


if __name__ == "__main__":
    main()


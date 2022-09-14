#! ./venv/bin/python
import torchvision
import torch as torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from RESNET.train import CifarData
from utils.helper import GetDevice, CleanCuda
import numpy as np


def ConfidenceMap(validationset, model):
    """
    param: validationset:

    param: model:
    """
    device = GetDevice()
    CleanCuda()
    model = model.to(device)
    model.eval()
    conf_map = {

    }
    _id = 1
    sm = nn.Softmax(dim=1)
    with torch.no_grad():
        for item in validationset:
            CleanCuda()
            img = item[0]
            label = item[1]
            img = img.to(device)
            scores = sm(model(img.unsqueeze(0))).squeeze()
            CleanCuda()
            guess = torch.argmax(scores).item()
            conf = scores[guess].item()
            match = (guess == label)
            conf_map[_id] = [conf, match]
            _id += 1
    return conf_map
        




def main():
    CleanCuda()
    cd = CifarData()
    _, valset = cd.get_dataset()
    resnet50 = models.resnet50(pretrained=False)
    resnet50.load_state_dict(torch.load('./weights/resnet50.pth'))
    cm = ConfidenceMap(valset, resnet50)
    np.save('dump/confidence_predictions.npy', cm)


if __name__ == "__main__":
    main()
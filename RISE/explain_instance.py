#!./venv/bin/python
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import resize
import numpy as np
import torchvision.datasets as datasets
import torchvision.models as models
from utils.helper import GetDevice, CleanCuda
from RESNET.train import CifarData
from torch import optim 
import torch
from RISE.explanations import generate_masks, explain
import torchvision.transforms as transforms
import argparse


def explain_instance():
    """
    """
    idx = args.idx
    with torch.no_grad():
        device = GetDevice()
        resnet50 = models.resnet50(pretrained=False).to(device)
        resnet50.load_state_dict(torch.load('./weights/resnet50.pth'))
        resnet50.eval()
        cd = CifarData()
        _, valset = cd.get_dataset()
        img = valset[idx][0].permute(1, 2, 0).cpu().detach().numpy()
        plt.imshow(img.astype('uint8'))
        img_tensor = valset[idx][0].permute(1, 2, 0).unsqueeze(0)
        print(len(valset))
        label = valset[idx][1] 
        dimensions = (32, 32)
        N = 2000
        s = 8
        p1 = 0.5
        masks = generate_masks(dimensions, N, s, p1)
        img = img_tensor.cpu().detach().numpy()
        print(img.shape)
        sal = explain(resnet50, img, masks, N, p1, 1)
        ans = sal[label]
        disp = img.copy().squeeze(0)
        plt.imshow(disp)
        plt.imshow(ans, cmap='jet', alpha=0.5)
        plt.colorbar()
        plt.savefig('rise_uncalibrated.png')


def main():
    explain_instance()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training params')
    parser.add_argument('--idx', type=int, default=0,
                        help='Validation set index')
    args = parser.parse_args()
    main()
#!./venv/bin/python
from lib2to3.pgen2 import grammar
import matplotlib.pyplot as plt
import numpy as np
from .grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from RESNET.train import CifarData
import torch as torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from utils.helper import GetDevice, CleanCuda

CIFAR_100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR_100_STD = [0.2675, 0.2565, 0.2761]

def cam_map():
    cd = CifarData()
    device = GetDevice()
    model = models.resnet50(pretrained=False).to(device)
    model.load_state_dict(torch.load('./weights/resnet50.pth'))
    model.eval()
    _, valset = cd.get_dataset()
    # target_layers = [model.layer4[-1]]
    target_layers = [model.conv1]
    input_tensor = valset[0][0].unsqueeze(0)
    label = valset[0][1]
    rgb_img = input_tensor.squeeze().permute(1, 2, 0).clone().cpu().detach().numpy()
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    targets = [ClassifierOutputTarget(label)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    print(f'MGV: {np.max(grayscale_cam)}')
    visualization = show_cam_on_image(img=rgb_img/np.max(rgb_img), mask=grayscale_cam, use_rgb=True, image_weight=0.50)
    plt.imshow(grayscale_cam)
    # plt.colorbar()
    # plt.imshow(rgb_img/np.max(rgb_img))
    plt.savefig('imgs/grad_cam.png')


def main():
    cam_map() 

if __name__ == "__main__":
    main()



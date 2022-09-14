#! ./venv/bin/python
import torchvision
import torch as torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from RESNET.train import CifarData
from RISE.explanations import generate_masks, explain
from utils.helper import GetDevice, CleanCuda
from SSIM.compute_ssim import compute_rise_ssim
import matplotlib.pyplot as plt
import numpy as np
from .temperature_scaling import ModelWithTemperature


def mean_ssim(uncalibrated_model, calibrated_model, valset):
    """
    param: uncalibrated_model: the model with no
    calibrated confidence scores

    param: calibrated_model: the model with
    calibrated confidence scores

    param: valset: the images in the validation set
    """
    device = GetDevice()
    sm  = nn.Softmax(dim = 1)
    dimensions = (32, 32)
    N = 2000
    s = 8
    p1 = 0.5
    masks = generate_masks(dimensions, N, s, p1)
    models = [uncalibrated_model.to(device), calibrated_model.to(device)]
    with torch.no_grad():
        for item in valset:
            CleanCuda()
            img  = item[0].permute(0, 2, 3, 1)
            label = item[1]
            img = img.to(device)
            img_ndarray = img.cpu().detach().numpy()
            disp = img_ndarray.copy().squeeze(0)
            for i,  model in enumerate(models):
                plt.clf()
                sal = explain(model, img_ndarray, masks, N, p1, 1)
                ans = sal[label]               
                plt.imshow(disp)
                plt.imshow(ans, cmap='jet', alpha=0.5)
                plt.colorbar()
                plt.savefig(f'imgs/rise_{i}.png')
            ssim = compute_rise_ssim('imgs/rise_0.png', 'imgs/rise_1.png')
    return 0



def main():
    CleanCuda()
    cd = CifarData()
    _, valset = cd.get_data_loader()
    _, valloader = cd.get_data_loader()
    uncalibrated_model = models.resnet50(pretrained=False)
    uncalibrated_model.load_state_dict(torch.load('./weights/resnet50.pth'))
    uncalibrated_model.eval()
    calibrated_model = ModelWithTemperature(uncalibrated_model)
    calibrated_model.set_temperature(valloader)
    calibrated_model.eval()
    mean_ssim(uncalibrated_model, calibrated_model, valset)

if __name__ == "__main__":
    main()
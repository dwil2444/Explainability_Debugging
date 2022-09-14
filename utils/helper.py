import torch as torch
import gc as gc
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import statistics
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

def CleanCuda():
    gc.collect()
    torch.cuda.empty_cache()


def GetDevice():
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        # print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device
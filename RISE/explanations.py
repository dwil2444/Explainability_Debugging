import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from tqdm import tqdm
from utils.helper import GetDevice

def generate_masks(input_size, N, s, p1):
    """
    param: input_size: tuple representing image
    dimensions (H, W)

    param: N: the number of 
    masks to generate

    param: s: the size of the 
    masks : [hxw]

    param: p1: the proportion
    of pixels in the image 
    to mask
    """
    np.random.seed(0)
    cell_size = np.ceil(np.array(input_size) / s)
    up_size = (s + 1) * cell_size

    grid = np.random.rand(N, s, s) < p1
    grid = grid.astype('float32')

    masks = np.empty((N, *input_size))

    for i in tqdm(range(N), desc='Generating masks'):
        # Random shifts
        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        # Linear upsampling and cropping
        masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                anti_aliasing=False)[x:x + input_size[0], y:y + input_size[1]]
    masks = masks.reshape(-1, *input_size, 1)
    return masks


def explain(model, inp, masks, N, p1, batch_size=1):
    """
    param: model: the neural architecture

    param: inp: the input image

    param: masks: the "masks" generated from
    the generate_masks function

    param: N: the number of 
    masks to generate

    param: p1: the proportion
    of pixels in the image 
    to mask

    """
    np.random.seed(0)
    device = GetDevice()
    sm = nn.Softmax(dim=1)
    input_size = (inp.shape[1], inp.shape[2])
    preds = []
    # Make sure multiplication is being done for correct axes
    masked = inp * masks
    masked = torch.tensor(masked).float().to(device)
    masked = masked.permute(0, 3, 1, 2)
    for i in tqdm(range(0, N, batch_size), desc='Explaining'):
        preds.append(sm(model(masked[i:min(i+batch_size, N)])).cpu().detach().numpy())
    preds = np.concatenate(preds)
    sal = preds.T.dot(masks.reshape(N, -1)).reshape(-1, *input_size)
    sal = sal / N / p1
    return sal
import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM


class GradCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(
            GradCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        """
        param: self:
        param: input_tensor:
        param: target_layer:
        param: target_category:
        param: activations:
        param: grads:
        """
        print(f'Shape of Gradients {grads.shape}')
        print(f'Neuron Importance Weights {np.mean(grads, axis=(2, 3)).shape}')
        return np.mean(grads, axis=(2, 3))

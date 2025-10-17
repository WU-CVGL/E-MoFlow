import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-9

def gradient(a: torch.Tensor):
    kernel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=a.dtype,
        device=a.device
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=a.dtype,
        device=a.device
    ).view(1, 1, 3, 3)
    
    channels = a.shape[1]
    kernel_x = kernel_x.repeat(channels, 1, 1, 1)
    kernel_y = kernel_y.repeat(channels, 1, 1, 1)
    
    grad_x = F.conv2d(a, kernel_x, padding=1, groups=channels)
    grad_y = F.conv2d(a, kernel_y, padding=1, groups=channels)
    return grad_x, grad_y

class FocusLoss(nn.Module):
    def __init__(self, loss_type='variance', norm='l2'):
        super().__init__()
        self.loss_type = loss_type
        self.norm = norm
        self._validate_parameters()
        
    def _validate_parameters(self):
        valid_loss_types = ['variance', 'gradient_magnitude']
        if self.loss_type not in valid_loss_types:
            raise ValueError(f"Invalid loss_type: {self.loss_type}. Must be one of {valid_loss_types}.")
        if self.loss_type == 'gradient_magnitude' and self.norm not in ['l1', 'l2']:
            raise ValueError(f"Invalid norm: {self.norm} for loss_type 'gradient_magnitude'. Use 'l1' or 'l2'.")
    
    def forward(self, iwes):
        if self.loss_type == 'variance':
            return self._image_variance(iwes)
        else:
            return self._gradient_magnitude(iwes)
    
    def _image_variance(self, iwes):
        variances = torch.var(iwes, dim=(1, 2))
        return torch.mean(variances)
    
    def _gradient_magnitude(self, iwes):
        if iwes.dim() == 3:
            iwes = iwes.unsqueeze(1)
        dx, dy = gradient(iwes)
        if self.norm == 'l2':
            mag = torch.square(dx) + torch.square(dy)
        elif self.norm == 'l1':
            mag = torch.abs(dx) + torch.abs(dy)
        else:
            raise ValueError(f"Invalid norm: {self.norm}")
        return torch.mean(mag)
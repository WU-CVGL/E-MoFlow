import torch
import kornia

from typing import Optional

class KorniaPixel2Cam:
    def __init__(self, H: int, W: int, device: torch.device):
        self.grid = kornia.utils.create_meshgrid(
            H, W, normalized_coordinates=False).to(device)
        self.grid = self.grid.squeeze(0)
        self.ones = torch.ones(H, W, 1, device=device)
        self.pixels_homogeneous = torch.cat(
            [self.grid, self.ones], dim=-1)  # [H,W,3]
        
    def __call__(self, K: torch.Tensor, depth: Optional[torch.Tensor] = None) -> torch.Tensor:
        if K.dim() == 2:
            K = K.unsqueeze(0)
        B = K.shape[0]
        
        pixels_batch = self.pixels_homogeneous.unsqueeze(0).expand(B, -1, -1, -1).to(K.device)
        
        K_4x4 = torch.zeros(B, 4, 4, device=K.device)
        K_4x4[:, :3, :3] = K
        K_4x4[:, 3, 3] = 1.0
        K_4x4_inv = torch.inverse(K_4x4)
        
        if depth is None:
            depth = torch.ones(B, 1, *self.grid.shape[:2], device=K.device)
            
        return kornia.geometry.camera.pixel2cam(
            depth, K_4x4_inv, pixels_batch)[..., :2]
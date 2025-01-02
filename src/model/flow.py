import torch 
import numpy as np

from src.model.inr import EventFlowINR

class DenseOpticalFlowCalc:
    def __init__(self, grid_size, model: EventFlowINR, normalize_coords=True, device="cuda"):
        self.H, self.W = grid_size
        x, y = np.meshgrid(np.arange(self.W), np.arange(self.H))
        x = torch.tensor(x, dtype=torch.float32).view(-1)
        y = torch.tensor(y, dtype=torch.float32).view(-1)
        if normalize_coords:
            x = x / (self.W - 1)
            y = y / (self.H - 1)
        self.x = x.to(device)
        self.y = y.to(device)

        self.model = model
        self.device = device

    def extract_flow_from_inr(self, t):
        num_points = self.x.shape[0]
        num_timesteps = t.shape[1]
        x_batch = self.x.repeat(num_timesteps)
        y_batch = self.y.repeat(num_timesteps)
        t_batch = t.view(-1).repeat_interleave(num_points).to(self.device)
        txy = torch.stack((t_batch, x_batch, y_batch), dim=1)
        
        with torch.no_grad():
            flow = self.model(txy)  # [num_points*num_timesteps, 2]
        
        u = flow[:, 0].view(num_timesteps, num_points)
        v = flow[:, 1].view(num_timesteps, num_points)
        U = u.view(num_timesteps, self.H, self.W)
        V = v.view(num_timesteps, self.H, self.W)
        
        U, V = U * (self.W - 1), V * (self.H - 1)
        
        return u, v, U, V
    
    def sparsify_flow(self, flow, sparse_ratio=0.1, threshold=1):
        H, W, C = flow.shape
        total_points = H * W
        keep_points = int(total_points * sparse_ratio)
        
        flow_flat = flow.view(-1, C)
        random_indices = torch.randperm(total_points, device=self.device)[:keep_points]
        sparse_flow = flow_flat[random_indices, :]
        
        if threshold is not None:
            flow_magnitude = torch.norm(sparse_flow[..., :2], dim=-1)  # [N]
            mask = flow_magnitude > threshold  # [N]
            valid_indices = torch.where(mask)[0]
            sparse_flow = sparse_flow[valid_indices, :]
            indices = random_indices[valid_indices]
        else:
            indices = random_indices
            
        return sparse_flow, indices
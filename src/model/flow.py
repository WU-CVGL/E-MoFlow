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
        
        U, V = U * (self.W - 1) * 52.63 / 908.72409, V * (self.H - 1) * 52.63 / 908.72409
        
        return u, v, U, V
    
    def sparsify_flow(self, flow, sparse_ratio=0.1, threshold=1):
        H, W, C = flow.shape  
        
        total_points = H * W
        keep_points = int(total_points * sparse_ratio)
        num_blocks_h = int(np.sqrt(keep_points))
        num_blocks_w = num_blocks_h
        
        block_h = H // num_blocks_h
        block_w = W // num_blocks_w
        
        num_blocks_h = H // block_h
        num_blocks_w = W // block_w
        
        flows_list = []
        indices_list = []
        
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                h_start = i * block_h
                h_end = (i + 1) * block_h
                w_start = j * block_w
                w_end = (j + 1) * block_w
                
                current_block = flow[h_start:h_end, w_start:w_end]  # [block_h, block_w, C]
                flow_magnitude = torch.norm(current_block[..., :2], dim=-1)  # [block_h, block_w]
                
                max_idx = flow_magnitude.view(-1).argmax()
                local_h = max_idx // block_w
                local_w = max_idx % block_w
                
                global_h = h_start + local_h
                global_w = w_start + local_w
                global_idx = global_h * W + global_w
                
                flows_list.append(flow[global_h, global_w])
                indices_list.append(global_idx)
        
        sparse_flow = torch.stack(flows_list)  # [N, C]
        indices = torch.tensor(indices_list, device=flow.device)  # [N]
        
        if threshold is not None:
            flow_magnitude = torch.norm(sparse_flow[..., :2], dim=-1)
            mask = flow_magnitude > threshold
            valid_indices = torch.where(mask)[0]
            sparse_flow = sparse_flow[valid_indices]
            indices = indices[valid_indices]
        
        return sparse_flow, indices
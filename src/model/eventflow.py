import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from src.model import embedder

class EventFlowINR(nn.Module):
    def __init__(self, config, D=12, W=256, input_ch=3, output_ch=2, skips=[2]):
        super().__init__()
        self.config = config
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.skips = skips
        # self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        # network
        self.coord_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        self.output_linear = nn.Linear(W, output_ch)

        for linear in self.coord_linears:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
        nn.init.xavier_uniform_(self.output_linear.weight)
        nn.init.zeros_(self.output_linear.bias)

    def forward(self, coord_txy: torch.Tensor):
        # create positional encoding
        embed_fn, input_ch = embedder.get_embedder(self.config)

        # forward positional encoding
        coord_xyt_flat = torch.reshape(coord_txy, [-1, coord_txy.shape[-1]])
        embedded_coord_xyt = embed_fn(coord_xyt_flat)
        # embedded_coord_xyt = coord_xyt_flat.clone()
        
        # input_pts, input_views = torch.split(embedded, [self.input_ch, self.input_ch_views], dim=-1)
        h = embedded_coord_xyt
        for i, l in enumerate(self.coord_linears):
            h = self.coord_linears[i](h)
            h = F.relu(h)
            # h = self.leaky_relu(h)
            if i in self.skips:
                h = torch.cat([embedded_coord_xyt, h], -1)
       
        outputs = self.output_linear(h)
        # outputs = torch.sigmoid(outputs)
        outputs = torch.reshape(outputs, list(coord_txy.shape[:-1]) + [outputs.shape[-1]])
        return outputs
    
class DenseOpticalFlowCalc:
    def __init__(
        self, 
        grid_size: List,
        intrinsic_mat: torch.Tensor, 
        model: EventFlowINR, 
        normalize_coords_mode="NORM_PLANE", 
        device="cuda"
    ):
        self.H, self.W = grid_size
        x, y = np.meshgrid(np.arange(self.W), np.arange(self.H))
        x = torch.tensor(x, dtype=torch.float32).view(-1)
        y = torch.tensor(y, dtype=torch.float32).view(-1)
        
        if normalize_coords_mode == "UV_SPACE":
            x = x / (self.W - 1)
            y = y / (self.H - 1)
        elif normalize_coords_mode == "NORM_PLANE":
            fx, fy = intrinsic_mat[0,0], intrinsic_mat[1,1]
            cx, cy = intrinsic_mat[0,2], intrinsic_mat[1,2]
            x = (x - cx) / fx
            y = (y - cy) / fy
        else:
            pass
        
        self.x = x.to(device)
        self.y = y.to(device)
        self.K = intrinsic_mat
        self.model = model
        self.device = device
        self.normalize_coords_mode = normalize_coords_mode

    # def extract_flow_from_inr(self, t, time_scale):
    #     num_points = self.x.shape[0]
    #     num_timesteps = t.shape[1]
    #     x_batch = self.x.repeat(num_timesteps)
    #     y_batch = self.y.repeat(num_timesteps)
    #     t_batch = t.view(-1).repeat_interleave(num_points).to(self.device)
    #     txy = torch.stack((t_batch, x_batch, y_batch), dim=1)
        
    #     with torch.no_grad():
    #         flow = self.model(txy)  # [num_points*num_timesteps, 2]
        
    #     u = flow[:, 0].view(num_timesteps, num_points)
    #     v = flow[:, 1].view(num_timesteps, num_points)
    #     U_norm = u.view(num_timesteps, self.H, self.W)
    #     V_norm = v.view(num_timesteps, self.H, self.W)
        
    #     if self.normalize_coords_mode == "UV_SPACE":
    #         U, V = U_norm * (self.W - 1) * time_scale, V_norm * (self.H - 1) * time_scale
    #     elif self.normalize_coords_mode == "NORM_PLANE":
    #         fx, fy = self.K[0,0], self.K[1,1]
    #         U, V = U_norm * fx * time_scale, V_norm * fy * time_scale
    #     else:
    #         U, V = U_norm * time_scale, V_norm * time_scale
    #     return U, V, U_norm, V_norm
    
    def extract_flow_from_inr(self, t, time_scale):
        num_points = self.x.shape[0]
        t_batch = t * torch.ones(num_points, device=self.device).to(self.device)
        txy = torch.stack((t_batch, self.x, self.y), dim=1).unsqueeze(0)
        
        with torch.no_grad():
            flow = self.model(txy)  # [1, num_points, 2]
        
        flow = flow.squeeze(0)
        u = flow[:, 0]
        v = flow[:, 1]
        U_norm = u.view(self.H, self.W)
        V_norm = v.view(self.H, self.W)
        
        if self.normalize_coords_mode == "UV_SPACE":
            U, V = U_norm * (self.W - 1) * time_scale, V_norm * (self.H - 1) * time_scale
        elif self.normalize_coords_mode == "NORM_PLANE":
            fx, fy = self.K[0,0], self.K[1,1]
            U, V = U_norm * fx * time_scale, V_norm * fy * time_scale
        else:
            U, V = U_norm * time_scale, V_norm * time_scale
        return U, V, U_norm, V_norm
    
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
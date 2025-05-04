import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from src.model import embedder
from torchdiffeq import odeint

class EventFlowINR(nn.Module):
    def __init__(self, config, D=8, W=256, input_ch=3, output_ch=2, skips=[4]):
        super().__init__()
        self.config = config
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.skips = skips
        # self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # network
        self.coord_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        self.output_linear = nn.Linear(W+input_ch if self.skips[-1]==D-1 else W, output_ch)

    def forward(self, coord_txy: torch.Tensor):
        # create positional encoding
        embed_fn, input_ch = embedder.get_embedder(self.config)

        # forward positional encoding
        coord_xyt_flat = torch.reshape(coord_txy, [-1, coord_txy.shape[-1]])
        embedded_coord_xyt = embed_fn(coord_xyt_flat)
        
        h = embedded_coord_xyt
        layer_outputs = []
        for i, l in enumerate(self.coord_linears):
            h = self.coord_linears[i](h)
            if len(layer_outputs) > 0:
                h = h + layer_outputs[-1]
            h = F.relu(h)

            layer_outputs.append(h)
            if i in self.skips:
                h = torch.cat([embedded_coord_xyt, h], -1)
       
        outputs = self.output_linear(h)
        outputs = torch.reshape(outputs, list(coord_txy.shape[:-1]) + [outputs.shape[-1]])
        return outputs
    
class ForwardFlowODEFunc(nn.Module):
    def __init__(self, nn_model, device):
        super(ForwardFlowODEFunc, self).__init__()
        self.nn_model = nn_model.to(device)

    def forward(self, t, state):
        batch_size = state.shape[0]  # H*W
        t_tensor = t * torch.ones(batch_size, 1).to(state.device)
        inputs = torch.cat([t_tensor, state], dim=1)
        uv = self.nn_model(inputs)
        return uv
    
class DenseFlowExtractor:
    def __init__(self, grid_size: List, device: torch.device):
        self.H, self.W = grid_size
        x, y = np.meshgrid(np.arange(self.W), np.arange(self.H))
        x = torch.tensor(x, dtype=torch.float32).view(-1)
        y = torch.tensor(y, dtype=torch.float32).view(-1)
        
        self.x = x.to(device)
        self.y = y.to(device)
        self.device = device
        
    def integrate_flow(self, model, t_start, t_end, num_steps=2):
        t = torch.linspace(t_start, t_end, num_steps).to(self.device)
        initial_positions = torch.stack((self.x, self.y), dim=1)
        
        odefunc = ForwardFlowODEFunc(model, self.device)
        solution = odeint(odefunc, initial_positions, t, method="euler", options={"step_size": 10})  # [num_steps, H*W, 2]
        final_positions = solution[-1]  # [H*W, 2]
        displacement = final_positions - initial_positions  # [H*W, 2]
        
        u, v = displacement[:, 0], displacement[:, 1]
        U = u.view(self.H, self.W)
        V = v.view(self.H, self.W)
        pred_flow = torch.stack((V, U), dim=2).permute(2, 0, 1)
        
        return pred_flow
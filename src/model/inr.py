import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import embedder
from typing import Dict

class EventFlowINR(nn.Module):
    def __init__(self, config, D=12, W=256, input_ch=3, output_ch=2, skips=[2]):
        super().__init__()
        self.config = config
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.skips = skips

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
            if i in self.skips:
                h = torch.cat([embedded_coord_xyt, h], -1)
       
        outputs = self.output_linear(h)
        # outputs = torch.sigmoid(outputs)
        outputs = torch.reshape(outputs, list(coord_txy.shape[:-1]) + [outputs.shape[-1]])
        return outputs
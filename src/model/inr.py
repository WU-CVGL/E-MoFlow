import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import embedder
from typing import Dict

class EventFlowINR(nn.Module):
    def __init__(self, config, D=8, W=256, input_ch=63, output_ch=2, skips=[4]):
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

# class NeRF(nn.Module):
#     def __init__(self, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=False,
#                  channels=3):
#         super().__init__()
#         self.D = D
#         self.W = W
#         self.input_ch = input_ch
#         self.input_ch_views = input_ch_views
#         self.skips = skips
#         self.use_viewdirs = use_viewdirs
#         self.channels = channels

#         # network
#         self.coord_linears = nn.ModuleList(
#             [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
#                                         range(D - 1)])

#         self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

#         if use_viewdirs:
#             self.feature_linear = nn.Linear(W, W)
#             self.alpha_linear = nn.Linear(W, 1)
#             self.rgb_linear = nn.Linear(W // 2, channels)
#         else:
#             self.output_linear = nn.Linear(W, channels + 1)

#     # positional encoding和nerf的mlp
#     def forward(self, iter_step, pts, viewdirs, args):
#         # create positional encoding
#         embed_fn, input_ch = embedder.get_embedder(args, args.multires, args.i_embed)
#         input_ch_views = 0
#         embeddirs_fn = None
#         if args.use_viewdirs:
#             embeddirs_fn, input_ch_views = embedder.get_embedder(args, args.multires_views, args.i_embed)
#         # forward positional encoding
#         pts_flat = torch.reshape(pts, [-1, pts.shape[-1]])
#         embedded = embed_fn(pts_flat)

#         if args.use_barf_c2f:
#             embedded = barf_c2f_weight(iter_step, embedded, input_ch, args)
#             embedded = torch.cat([pts_flat, embedded], -1)  # [..., 63]

#         if viewdirs is not None:
#             # embedded_dirs:[1024x64, 27]
#             input_dirs = viewdirs[:, None].expand(pts.shape)
#             input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
#             embedded_dirs = embeddirs_fn(input_dirs_flat)
#             if args.use_barf_c2f:
#                 embedded_dirs = barf_c2f_weight(iter_step, embedded_dirs, input_ch_views, args)
#                 embedded_dirs = torch.cat([input_dirs_flat, embedded_dirs], -1)  # [..., 27]
#             embedded = torch.cat([embedded, embedded_dirs], -1)

#         input_pts, input_views = torch.split(embedded, [self.input_ch, self.input_ch_views], dim=-1)
#         h = input_pts
#         for i, l in enumerate(self.coord_linears):
#             h = self.coord_linears[i](h)
#             h = F.relu(h)
#             if i in self.skips:
#                 h = torch.cat([input_pts, h], -1)

#         if self.use_viewdirs:
#             alpha = self.alpha_linear(h)
#             feature = self.feature_linear(h)
#             h = torch.cat([feature, input_views], -1)

#             for i, l in enumerate(self.views_linears):
#                 h = self.views_linears[i](h)
#                 h = F.relu(h)

#             rgb = self.rgb_linear(h)
#             outputs = torch.cat([rgb, alpha], -1)  # [N, 4(RGBA A is sigma)]
#         else:
#             outputs = self.output_linear(h)

#         outputs = torch.reshape(outputs, list(pts.shape[:-1]) + [outputs.shape[-1]])

#         return outputs

#     def raw2output(self, crf_func, enable_crf: bool, sensor_type, raw, z_vals, rays_d, raw_noise_std=1.0):
#         raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

#         dists = z_vals[..., 1:] - z_vals[..., :-1]
#         dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)

#         dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

#         rgb = torch.sigmoid(raw[..., :self.channels])
#         # if enable_crf == True:
#         #     rgb = crf_func(rgb, sensor_type)
#         #     rgb = torch.sigmoid(rgb)
#         # elif enable_crf == False:
#         #     rgb = torch.exp(rgb)

#         noise = 0.
#         if raw_noise_std > 0.:
#             noise = torch.randn(raw[..., self.channels].shape) * raw_noise_std

#         alpha = raw2alpha(raw[..., self.channels] + noise, dists)
#         weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:,
#                           :-1]
#         rgb_map = torch.sum(weights[..., None] * rgb, -2)

#         depth_map = torch.sum(weights * z_vals, -1)
#         disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
#         acc_map = torch.sum(weights, -1)

#         sigma = F.relu(raw[..., self.channels] + noise)

#         return rgb_map, disp_map, acc_map, weights, depth_map, sigma
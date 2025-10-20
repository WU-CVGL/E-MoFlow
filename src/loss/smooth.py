import torch
import torch.nn as nn
import torch.nn.functional as F


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


class FlowSmoothnessLoss(nn.Module):
    """
    Flow smoothness loss: L_smooth = sum_{x,y} (|∇_x V(x,y)| + |∇_y V(x,y)|)

    This loss encourages the flow field to be spatially smooth by penalizing
    the L1 norm of spatial gradients.
    """
    def __init__(self):
        super().__init__()

    def forward(self, flow):
        if flow.dim() == 3 and flow.shape[-1] == 2:
            raise ValueError("Cannot compute smoothness loss on sparse samples. "
                           "Flow must be dense with shape [B, 2, H, W]")

        # flow should be [B, 2, H, W]
        if flow.dim() == 4 and flow.shape[1] == 2:
            dx, dy = gradient(flow)  # dx, dy: [B, 2, H, W]
            # Sum absolute gradients: |∇_x V| + |∇_y V|
            smoothness = torch.abs(dx) + torch.abs(dy)
            return torch.mean(smoothness)
        else:
            raise ValueError(f"Expected flow shape [B, 2, H, W], got {flow.shape}")


class SparseFlowSmoothnessLoss(nn.Module):
    """
    Sparse flow smoothness loss computed on sampled coordinates.
    Approximates smoothness by computing flow differences at neighboring sample points.
    """
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, sample_coords, sample_flow, flow_field):
        B, N, _ = sample_coords.shape

        # Create small offsets for finite difference (in pixel space)
        delta = 1.0  # 1 pixel offset
        offsets = torch.zeros(B, N, 2, 3, device=sample_coords.device, dtype=sample_coords.dtype)
        offsets[:, :, 0, :] = sample_coords  # right offset base
        offsets[:, :, 0, 1] += delta          # x + delta
        offsets[:, :, 1, :] = sample_coords  # down offset base
        offsets[:, :, 1, 2] += delta          # y + delta

        # query neighbor flow: [B, N*2, 3] -> [B, N*2, 2]
        neighbor_coords = offsets.reshape(B, N * 2, 3)
        neighbor_flow = flow_field(neighbor_coords) 
        flow_right = neighbor_flow[:, :N, :]
        flow_down = neighbor_flow[:, N:, :]

        # Compute finite differences (delta=1.0)
        # ∂f/∂x ≈ f(x+1) - f(x), ∂f/∂y ≈ f(y+1) - f(y)
        grad_x = flow_right - sample_flow
        grad_y = flow_down - sample_flow

        # L1 norm of gradients: |∇_x V| + |∇_y V|
        smoothness = torch.abs(grad_x) + torch.abs(grad_y)

        return torch.mean(smoothness)
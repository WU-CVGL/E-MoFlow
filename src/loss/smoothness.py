import torch
import numpy as np
from scipy.spatial import cKDTree

def chebyshev_distance(a, b):
    """calculate chebyshev distance"""
    return np.max(np.abs(a - b))

def manhattan_distance(a, b):
    """calculate manhattan distance"""
    return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])

def calculate_local_spatial_smoothness_loss(
        events: torch.Tensor, flows: torch.Tensor, search_radius):
    """
    Use KD-tree to find neighboring event points and compute local spatial smoothness loss.

    Args:
        events (torch.Tensor): tensor n*4(x,y,t,p)
        flows (torch.Tensor): n*2的tensor，包含每个事件的光流u和v。
        search_radius (float): 搜索邻域的半径。

    Returns:
        loss (torch.Tensor): 约束光流方向的一致性损失。
    """


    xy_coords = events[:, :2].numpy()  # 转换为numpy以使用KDTree
    kdtree = cKDTree(xy_coords)

    loss = 0.0
    n = events.shape[0]

    for i in range(n):
        neighbors_indices = kdtree.query_ball_point(xy_coords[i], search_radius)
        
        current_flow = flows[i]
        
        for idx in neighbors_indices:
            if idx != i:  
                neighbor_flow = flows[idx]
                
                direction_diff = (current_flow - neighbor_flow).norm()
                loss += direction_diff

    return loss
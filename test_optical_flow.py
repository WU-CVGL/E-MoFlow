import torch
import numpy as np
import imageio.v2 as imageio

from pathlib import Path
from src.utils.visualizer import Visualizer

# imageio.plugins.freeimage.download()

def load_optical_flow(flow_path):
    """Load optical flow map and return the horizontal and vertical component tensors"""
    flow_16bit = imageio.imread(flow_path, format='PNG-FI').astype(np.float32)
    
    u_component = (flow_16bit[..., 0] - 2**15) / 128.0  # x
    v_component = (flow_16bit[..., 1] - 2**15) / 128.0  # y
    valid = flow_16bit[..., 2].astype(bool)
    
    U = torch.from_numpy(u_component)
    V = torch.from_numpy(v_component)
    mask = torch.from_numpy(valid)
    
    return U, V, mask

def save_flow(file_path: Path, flow: np.ndarray):
    """Save the optical flow as a 16-bit PNG."""
    height, width = flow.shape[0], flow.shape[1]
    flow_16bit = np.zeros((height, width, 3), dtype=np.uint16)
    flow_16bit[..., 0] = (flow[..., 0] * 128 + 2**15).astype(np.uint16)  # y-component
    flow_16bit[..., 1] = (flow[..., 1] * 128 + 2**15).astype(np.uint16)  # x-component
    flow_16bit[..., 2] = 1
    imageio.imwrite(str(file_path), flow_16bit, format='PNG-FI')

if __name__ == "__main__":
    
    viz = Visualizer(
        image_shape=(480, 640),
        show=True,
        save=True,
        save_dir="./outputs/flowmap",
    )
    # 使用示例
    flow_path = "./000046.png"  # 替换为实际路径
    
    try:
        U, V, mask = load_optical_flow(flow_path)
        mask_float = mask.float()
        # U, V = U * mask_float, V * mask_float
        flowmap = torch.stack([U, V], dim=2).numpy()
        save_flow("./outputs/flowmap/000020_rewrite.png", flowmap)
        
        print(f"U tensor shape: {U.shape}")  # 应为 (H, W)
        print(f"V tensor shape: {V.shape}")  # 应为 (H, W)
        print(f"Max horizontal flow: {U.max().item():.2f}")
        print(f"Min vertical flow: {V.min().item():.2f}")
        
        eval_color_flow, wheel = viz.visualize_optical_flow(
            flow_x=U.cpu().numpy(),
            flow_y=V.cpu().numpy(),
            visualize_color_wheel=False,
            file_prefix="gt_color_flow",
            save_flow=False,
            ord=0.5,
        )
        
    except Exception as e:
        print(f"Error loading optical flow: {str(e)}")
    

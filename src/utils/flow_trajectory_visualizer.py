import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional, Tuple

from src.utils.logger import get_logger
logger = get_logger()


def visualize_flow_trajectories_3d(
    flow_sequence: List[np.ndarray],
    time_sequence: List[float],
    save_path: str,
    image_shape: Tuple[int, int],
    sampling_stride: int = 20,
    max_trajectories: Optional[int] = 1000,
    alpha: float = 0.6,
    line_width: float = 1.0,
    figure_size: Tuple[int, int] = (16, 12),
    elev: float = 30,
    azim: float = 45,
    view_init_list: Optional[List[Tuple[float, float]]] = None
):
    logger.info(f"Visualizing XYT 3D optical flow trajectories...")

    if len(flow_sequence) != len(time_sequence):
        raise ValueError(f"assertion error: len(flow_sequence) != len(time_sequence)")

    if len(flow_sequence) == 0:
        raise ValueError("flow sequence is empty")

    H, W = image_shape
    n_frames = len(flow_sequence)

    y_coords = np.arange(0, H, sampling_stride)
    x_coords = np.arange(0, W, sampling_stride)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
    sample_points = np.stack([yy.flatten(), xx.flatten()], axis=1)  # [N, 2] (y, x)

    logger.info(f"sample points number: {len(sample_points)}")
    if max_trajectories is not None and len(sample_points) > max_trajectories:
        indices = np.random.choice(len(sample_points), max_trajectories, replace=False)
        sample_points = sample_points[indices]
        logger.info(f"sample points number: {len(sample_points)}")

    # calc 3d trajectories
    trajectories = []
    trajectory_colors = []
    for idx, (start_y, start_x) in enumerate(sample_points):
        trajectory = [[start_x, start_y, time_sequence[0]]]

        accumulated_flow_x = 0
        accumulated_flow_y = 0

        current_x = start_x
        current_y = start_y

        # accumulate flow
        for i in range(n_frames):
            flow = flow_sequence[i]  # [2, H, W]
            y_idx = int(np.clip(current_y, 0, H - 1))
            x_idx = int(np.clip(current_x, 0, W - 1))
            flow_x = flow[0, y_idx, x_idx]  
            flow_y = flow[1, y_idx, x_idx]

            current_x += flow_x
            current_y += flow_y
            accumulated_flow_x += flow_x
            accumulated_flow_y += flow_y

            # add to traj
            if i < n_frames - 1:
                trajectory.append([current_x, current_y, time_sequence[i + 1]])

        trajectories.append(np.array(trajectory))

        # assign colors to traj
        avg_flow_x = accumulated_flow_x / n_frames
        avg_flow_y = accumulated_flow_y / n_frames
        angle = np.arctan2(avg_flow_y, avg_flow_x) 
        angle_normalized = (angle + np.pi) / (2 * np.pi)
        magnitude = np.sqrt(avg_flow_x**2 + avg_flow_y**2)

        trajectory_colors.append((angle_normalized, magnitude))

    logger.info(f"successfully calculated {len(trajectories)} trajectories")

    # norm color
    magnitudes = np.array([c[1] for c in trajectory_colors])
    max_magnitude = np.percentile(magnitudes, 95)  
    if max_magnitude == 0:
        max_magnitude = 1.0

    # create figure
    if view_init_list is None:
        view_init_list = [(elev, azim)]

    for view_idx, (elev_angle, azim_angle) in enumerate(view_init_list):
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(111, projection='3d')

        # plot traj.
        for trajectory, (angle_norm, magnitude) in zip(trajectories, trajectory_colors):
            # HSV: Hue=angle, Saturation=1, Value=magnitude
            normalized_mag = np.clip(magnitude / max_magnitude, 0, 1)
            color = plt.cm.hsv(angle_norm)

            color = np.array(color)
            color[:3] = color[:3] * (0.3 + 0.7 * normalized_mag)  

            ax.plot(
                trajectory[:, 0],  # X
                trajectory[:, 1],  # Y
                trajectory[:, 2],  # T
                color=color,
                alpha=alpha,
                linewidth=line_width
            )

        ax.set_xlabel('Image X (pixels)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Image Y (pixels)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Time (s)', fontsize=12, fontweight='bold')
        ax.set_title('3D Optical Flow Trajectories (XYT)',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_zlim(time_sequence[0], time_sequence[-1])
        ax.view_init(elev=elev_angle, azim=azim_angle)
        sm = cm.ScalarMappable(cmap=cm.hsv, norm=plt.Normalize(vmin=0, vmax=2*np.pi))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.6, aspect=20)
        cbar.set_label('Flow Direction (rad)', fontsize=10, fontweight='bold')
        cbar.set_ticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        cbar.set_ticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

        # save figure
        if len(view_init_list) > 1:
            save_path_with_view = save_path.replace('.png', f'_view{view_idx}_elev{int(elev_angle)}_azim{int(azim_angle)}.png')
        else:
            save_path_with_view = save_path

        plt.tight_layout()
        plt.savefig(save_path_with_view, dpi=150, bbox_inches='tight')
        logger.info(f"3D optical flow field figure saved to {save_path_with_view}")
        plt.close(fig)


def collect_and_visualize_flow_trajectories(
    flow_list: List[torch.Tensor],
    time_list: List[float],
    save_path: str,
    image_shape: Tuple[int, int],
    sampling_stride: int = 20,
    max_trajectories: Optional[int] = 1000,
    multiple_views: bool = True
):
    flow_sequence = []
    for flow in flow_list:
        if isinstance(flow, torch.Tensor):
            flow_np = flow.detach().cpu().numpy()
        else:
            flow_np = flow

        if flow_np.ndim == 4:  # [B, 2, H, W]
            flow_np = flow_np[0] 

        flow_sequence.append(flow_np)

    # set multiple views
    if multiple_views:
        view_init_list = [
            (30, 45),    
            (60, 45),    
            (15, 135),  
            (45, 0),     
        ]
    else:
        view_init_list = None

    visualize_flow_trajectories_3d(
        flow_sequence=flow_sequence,
        time_sequence=time_list,
        save_path=save_path,
        image_shape=image_shape,
        sampling_stride=sampling_stride,
        max_trajectories=max_trajectories,
        view_init_list=view_init_list
    )

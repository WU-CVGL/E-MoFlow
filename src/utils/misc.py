import os
import torch
import random
import logging
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Union

logger = logging.getLogger(__name__)

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

def fix_random_seed(seed_idx=0) -> None:
    random.seed(seed_idx)
    np.random.seed(seed_idx)
    torch.manual_seed(seed_idx)
    torch.random.manual_seed(seed_idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_idx)
        torch.cuda.manual_seed_all(seed_idx)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(0)

def load_camera_intrinsic(
    file_path: str
):
    K = np.loadtxt(file_path)
    K_tensor = torch.from_numpy(K).float()
    return K_tensor

def load_camera_pose(
    file_path: str
):
    camera_pose = np.loadtxt(file_path)
    camera_pose_tensor = torch.from_numpy(camera_pose).float()
    return camera_pose_tensor

def load_time_stamps(
    file_path: str
):
    timestamps = np.loadtxt(file_path)[:,0]
    timestamps_tensor = torch.from_numpy(timestamps).float()
    return timestamps_tensor

def get_sorted_txt_paths(folder_path: str) -> list:
    """
    Get sorted list of txt file paths from a folder.
    
    Args:
        folder_path (str): Path to the folder containing txt files
        
    Returns:
        list: Sorted list of txt file paths
    """
    # Get all txt files
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    # Sort files based on the numeric value in filename
    txt_files.sort(key=lambda x: int(x.split('.')[0]))
    
    # Create full paths
    txt_paths = [os.path.join(folder_path, f) for f in txt_files]
    
    return txt_paths

def get_filenames(file_list: List[str], indices: Union[List[int], int]) -> List[str]:
    """
    Retrieve filenames from a list based on provided indices.
    
    Args:
        file_list (List[str]): List of filenames
        indices (Union[List[int], int]): Single index or list of two indices [start, end]
        
    Returns:
        List[str]: List of retrieved filenames
        
    Examples:
        >>> files = ['file1.txt', 'file2.txt', 'file3.txt', 'file4.txt']
        >>> get_filenames(files, [0, 2])  # Get files from index 0 to 2
        ['file1.txt', 'file2.txt', 'file3.txt']
        >>> get_filenames(files, 1)  # Get file at index 1
        ['file2.txt']
    """
    try:
        # Handle single index case
        if isinstance(indices, int):
            return [file_list[indices]]
        # Handle range case
        elif len(indices) == 2:
            start, end = indices
            return file_list[start:end + 1]  # +1 to include the end index
        else:
            raise ValueError("Indices must be either a single integer or a list of two integers [start, end]")
    except IndexError as e:
        raise IndexError(f"Index out of range. List length is {len(file_list)}")
    
def check_file_utils(filename: str) -> bool:
    """Return True if the file exists.

    Args:
        filename (str): _description_

    Returns:
        bool: _description_
    """
    logger.debug(f"Check {filename}")
    res = os.path.exists(filename)
    if not res:
        logger.warning(f"{filename} does not exist!")
    return res


def check_key_and_bool(config: dict, key: str) -> bool:
    """Check the existance of the key and if it's True

    Args:
        config (dict): dict.
        key (str): Key name to be checked.

    Returns:
        bool: Return True only if the key exists in the dict and its value is True.
            Otherwise returns False.
    """
    return key in config.keys() and config[key]

def save_flow_error_as_text(
    flow_error_dict: dict, dir: str, fname: str = "metric.txt"
):
    save_file_name = os.path.join(dir, fname)
    with open(save_file_name, "a") as f:
        f.write(str(flow_error_dict) + "\n")
        
def save_time_log_as_text(
    time_stats_dict: dict, dir: str, fname: str = "time_stats.txt"
):
    save_file_name = os.path.join(dir, fname)
    with open(save_file_name, "a") as f:
        f.write(str(time_stats_dict) + "\n")
        
def save_theseus_result_as_text(
    theseus_result: dict, dir: str, fname: str = "theseus_result.txt"
):
    save_file_name = os.path.join(dir, fname)
    with open(save_file_name, "a") as f:
        f.write(str(theseus_result) + "\n")

def plot_velocity(lin_vel_array, ang_vel_array, save_dir, prefix_filename):

    os.makedirs(save_dir, exist_ok=True)

    def _plot_single(vel_array, plot_type):
        t = vel_array[:, 0]
        x = vel_array[:, 1]
        y = vel_array[:, 2]
        z = vel_array[:, 3]

        plt.figure(figsize=(10, 6))
        plt.plot(t, x, label='X', color='red', linewidth=1)
        plt.plot(t, y, label='Y', color='green', linewidth=1)
        plt.plot(t, z, label='Z', color='blue', linewidth=1)
        
        if plot_type == "linear":
            plt.ylabel("Linear Velocity", fontsize=12)
            plt.title("Linear Velocity Components vs Time", fontsize=14)
            filename = prefix_filename + "linear_velocity.png"
        else:
            plt.ylabel("Angular Velocity", fontsize=12)
            plt.title("Angular Velocity Components vs Time", fontsize=14)
            filename = prefix_filename + "angular_velocity.png"
            
        plt.xlabel("Time", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    _plot_single(lin_vel_array, "linear")
    _plot_single(ang_vel_array, "angular")
    
def visualize_velocities(
    gt_lin_vel, gt_ang_vel,
    eval_lin_vel, eval_ang_vel,
    t_eval
):
    # tensor2array
    def to_numpy(tensor):
        return tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
    
    gt_lin = to_numpy(gt_lin_vel)
    gt_ang = to_numpy(gt_ang_vel)
    eval_lin = to_numpy(eval_lin_vel)
    eval_ang = to_numpy(eval_ang_vel)
    time = to_numpy(t_eval).squeeze()
    
    # assert dim
    N = gt_lin.shape[0]
    assert time.ndim == 1, "t_eval must be 1-dimensional"
    assert N == time.shape[0], f"Time dimension mismatch: t_eval has {time.shape[0]} elements but velocities have {N}"
    
    import warnings
    warnings.filterwarnings("ignore", message="Blended transforms not yet supported")
    
    plot_config = {
        'lw': 1.2,
        'alpha': 0.8,
    }
    
    # plot linear velocities
    fig_lin, axs_lin = plt.subplots(3, 1, figsize=(12, 12))
    components = ['X', 'Y', 'Z']
    
    for i, (ax, comp) in enumerate(zip(axs_lin, components)):
        ax.plot(time, gt_lin[:, i], label='Ground Truth', color='#1f77b4', **plot_config)
        ax.plot(time, eval_lin[:, i], label='Estimated', color='#ff7f0e', linestyle='--', **plot_config)
        ax.set_title(f'Linear Velocity - {comp} Component', fontsize=12)
        if(i==2):
            ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Velocity (m/s)', fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        # ax.set_ylim(-1.0,1.0)
    
    # plt.tight_layout()
    fig_lin.suptitle("Linear Velocity Comparison", y=1.02, fontsize=14)
    
    # plot angular velocities
    fig_ang, axs_ang = plt.subplots(3, 1, figsize=(12, 12))
    
    for i, (ax, comp) in enumerate(zip(axs_ang, components)):
        ax.plot(time, gt_ang[:, i], label='Ground Truth', color='#2ca02c', **plot_config)
        ax.plot(time, eval_ang[:, i], label='Estimated', color='#d62728', linestyle='--', **plot_config)
        ax.set_title(f'Angular Velocity - {comp} Component', fontsize=12)
        if(i==2):
            ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Velocity (rad/s)', fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.0,1.0)
    
    # plt.tight_layout()
    fig_ang.suptitle("Angular Velocity Comparison", y=1.02, fontsize=14) 
    
    return fig_lin, fig_ang

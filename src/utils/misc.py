import os
import torch
import random
import logging
import numpy as np
import imageio.v2 as imageio

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

def fix_random_seed(seed_idx=42) -> None:
    random.seed(seed_idx)
    np.random.seed(seed_idx)
    torch.manual_seed(seed_idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_idx)
        torch.cuda.manual_seed_all(seed_idx)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
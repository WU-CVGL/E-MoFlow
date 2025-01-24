import torch
import random
import numpy as np
from typing import List, Tuple, Union

def fix_random_seed(seed_idx=666) -> None:
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
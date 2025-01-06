import torch
import numpy as np
from typing import List, Tuple, Union

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

def process_events(
    origin_events: torch.Tensor,
    image_size: Tuple[int, int],
    start_end: Tuple[float, float],
    normalize_coords: bool = True,
    normalize_time: bool = True
) -> torch.Tensor:
    """
    Load and process event data files, converting them into normalized tensor vectors.
    
    Args:
        events (torch.Tensor): original event data
        image_size (Tuple[int, int]): Target image size as (height, width)
        start_end Tuple[float, float]: the start and end timestamps of whole sequence
        normalize_coords (bool): Whether to normalize spatial coordinates
        normalize_time (bool): Whether to normalize timestamps
    
    Returns:
        torch.Tensor: Processed event data as a tensor with shape (N, 4) where N is 
                   the total number of valid events and columns are (t, x, y, p)
    """

    # Mask and sort
    mask = (0 <= origin_events[:, 1]) & (origin_events[:, 1] < image_size[1]) & \
            (0 <= origin_events[:, 2]) & (origin_events[:, 2] < image_size[0])
    events = origin_events[mask]
    events_sorted, sort_indices = torch.sort(events[:, 0])
    events_sorted = events[sort_indices]

    # Extract components
    timestamps = events_sorted[:, 0]
    x_coords = events_sorted[:, 1]
    y_coords = events_sorted[:, 2]
    polarities = events_sorted[:, 3]
    
    # Normalize timestamps if requested
    if normalize_time:
        norm_timestamps = (timestamps - start_end[0]) / (start_end[1] - start_end[0])
    
    # Normalize coordinates if requested
    if normalize_coords:
        norm_x_coords = x_coords / (image_size[1] - 1)  # Width
        norm_y_coords = y_coords / (image_size[0] - 1)  # Height
    
    # Combine back into tensor
    processed_events = torch.stack([norm_timestamps, norm_x_coords, norm_y_coords, polarities], axis=1)
    return processed_events

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
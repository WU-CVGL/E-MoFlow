import cv2
import torch
import imageio
import numpy as np

from pathlib import Path

def pixel_to_normalized_coords(pixel_coords, intrinsic_matrix):
    """
    Convert pixel coordinates to normalized coordinates.

    Args:
        pixel_coords: (B, N, 3) tensor of pixel coordinates
        intrinsic_matrix: (3, 3) or (B, 3, 3) tensor of intrinsic matrices

    Returns:
        normalized_coords: (B, N, 3) tensor of normalized coordinates
    """
    assert pixel_coords.dim() == 3 and pixel_coords.shape[2] == 3, \
        f"pixel_coords shape should be (B, N, 3), got {pixel_coords.shape}"
    assert intrinsic_matrix.shape == (3, 3) or \
           (intrinsic_matrix.dim() == 3 and intrinsic_matrix.shape[1:] == (3, 3)), \
        f"intrinsic_matrix shape should be (3, 3) or (B, 3, 3), got {intrinsic_matrix.shape}"

    if intrinsic_matrix.dim() == 2:
        intrinsic_matrix = intrinsic_matrix.unsqueeze(0) # (3, 3) -> (1, 3, 3)
    
    # same device
    device = pixel_coords.device
    if intrinsic_matrix.device != device:
        intrinsic_matrix = intrinsic_matrix.to(device)

    # intrinsic parameters (B, )
    fx = intrinsic_matrix[:, 0, 0]
    fy = intrinsic_matrix[:, 1, 1]
    cx = intrinsic_matrix[:, 0, 2]
    cy = intrinsic_matrix[:, 1, 2]

    # pixel coordinates (B, N)
    x = pixel_coords[:, :, 0]
    y = pixel_coords[:, :, 1]

    # Normalize
    x_n = (x - cx.unsqueeze(1)) / fx.unsqueeze(1)
    y_n = (y - cy.unsqueeze(1)) / fy.unsqueeze(1)

    # Stack to create normalized coordinates: (B, N, 3)
    ones = torch.ones_like(x_n)
    normalized_coords = torch.stack([x_n, y_n, ones], dim=2)

    return normalized_coords

def flow_to_normalized_coords(image_plane_flow, intrinsic_matrix):
    """
    Convert image plane flow to normalized coordinates.

    Args:
        image_plane_flow: (B, N, 3) tensor of image plane flow
        intrinsic_matrix: (3, 3) or (B, 3, 3) tensor of intrinsic matrices

    Returns:
        normalized_plane_flow: (B, N, 3) tensor of normalized plane flow
    """
    assert image_plane_flow.dim() == 3 and image_plane_flow.shape[2] == 3, \
        f"image_plane_flow shape should be (B, N, 3), got {image_plane_flow.shape}"
    assert intrinsic_matrix.shape == (3, 3) or \
           (intrinsic_matrix.dim() == 3 and intrinsic_matrix.shape[1:] == (3, 3)), \
        f"intrinsic_matrix shape should be (3, 3) or (B, 3, 3), got {intrinsic_matrix.shape}"

    if intrinsic_matrix.dim() == 2:
        intrinsic_matrix = intrinsic_matrix.unsqueeze(0) # (3, 3) -> (1, 3, 3)
    
    # same device
    device = image_plane_flow.device
    if intrinsic_matrix.device != device:
        intrinsic_matrix = intrinsic_matrix.to(device)

    # intrinsic parameters (B, )
    fx = intrinsic_matrix[:, 0, 0]
    fy = intrinsic_matrix[:, 1, 1]

    # flow (B, N)
    flow_x = image_plane_flow[:, :, 0]
    flow_y = image_plane_flow[:, :, 1]

    # Normalize (B, N)
    flow_x_n = flow_x / fx.unsqueeze(1)
    flow_y_n = flow_y / fy.unsqueeze(1)

    # Stack to create normalized flow: (B, N, 3)
    zeros = torch.zeros_like(flow_x_n)
    normalized_plane_flow = torch.stack([flow_x_n, flow_y_n, zeros], dim=2)

    return normalized_plane_flow

# The below code is coming from Zhu. et al, EV-FlowNet
# Optical flow loader
def estimate_corresponding_gt_flow(x_flow_in, y_flow_in, gt_timestamps, start_time, end_time):
    """Code obtained from https://github.com/daniilidis-group/EV-FlowNet

    The ground truth flow maps are not time synchronized with the grayscale images. Therefore, we
    need to propagate the ground truth flow over the time between two images.
    This function assumes that the ground truth flow is in terms of pixel displacement, not velocity.
    Pseudo code for this process is as follows:
    x_orig = range(cols)
    y_orig = range(rows)
    x_prop = x_orig
    y_prop = y_orig
    Find all GT flows that fit in [image_timestamp, image_timestamp+image_dt].
    for all of these flows:
    x_prop = x_prop + gt_flow_x(x_prop, y_prop)
    y_prop = y_prop + gt_flow_y(x_prop, y_prop)
    The final flow, then, is x_prop - x-orig, y_prop - y_orig.
    Note that this is flow in terms of pixel displacement, with units of pixels, not pixel velocity.

    Args:
        x_flow_in, y_flow_in - list of numpy arrays, each array corresponds to per pixel flow at
            each timestamp.
        gt_timestamps - timestamp for each flow array.
        start_time, end_time - gt flow will be estimated between start_time and end time.
    Returns:
        (x_disp, y_disp) ... Each displacement of x and y.
    """
    # Each gt flow at timestamp gt_timestamps[gt_iter] represents the displacement between
    # gt_iter and gt_iter+1.
    gt_iter = np.searchsorted(gt_timestamps, start_time, side="right") - 1
    gt_dt = gt_timestamps[gt_iter + 1] - gt_timestamps[gt_iter]
    x_flow = np.squeeze(x_flow_in[gt_iter, ...])
    y_flow = np.squeeze(y_flow_in[gt_iter, ...])
    dt = end_time - start_time

    # No need to propagate if the desired dt is shorter than the time between gt timestamps.
    if gt_dt >= dt:
        return x_flow * dt / gt_dt, y_flow * dt / gt_dt

    x_indices, y_indices = np.meshgrid(np.arange(x_flow.shape[1]), np.arange(x_flow.shape[0]))
    x_indices = x_indices.astype(np.float32)
    y_indices = y_indices.astype(np.float32)

    orig_x_indices = np.copy(x_indices)
    orig_y_indices = np.copy(y_indices)

    # Mask keeps track of the points that leave the image, and zeros out the flow afterwards.
    x_mask = np.ones(x_indices.shape, dtype=bool)
    y_mask = np.ones(y_indices.shape, dtype=bool)

    scale_factor = (gt_timestamps[gt_iter + 1] - start_time) / gt_dt
    total_dt = gt_timestamps[gt_iter + 1] - start_time

    prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor=scale_factor)

    gt_iter += 1
    while gt_timestamps[gt_iter + 1] < end_time:
        x_flow = np.squeeze(x_flow_in[gt_iter, ...])
        y_flow = np.squeeze(y_flow_in[gt_iter, ...])

        prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask)
        total_dt += gt_timestamps[gt_iter + 1] - gt_timestamps[gt_iter]
        gt_iter += 1

    final_dt = end_time - gt_timestamps[gt_iter]
    total_dt += final_dt

    final_gt_dt = gt_timestamps[gt_iter + 1] - gt_timestamps[gt_iter]
    x_flow = np.squeeze(x_flow_in[gt_iter, ...])
    y_flow = np.squeeze(y_flow_in[gt_iter, ...])
    scale_factor = final_dt / final_gt_dt

    prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor)
    x_shift = x_indices - orig_x_indices
    y_shift = y_indices - orig_y_indices
    x_shift[~x_mask] = 0
    y_shift[~y_mask] = 0
    return x_shift, y_shift


def prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor=1.0):
    """Code obtained from https://github.com/daniilidis-group/EV-FlowNet

    Propagates x_indices and y_indices by their flow, as defined in x_flow, y_flow.
    x_mask and y_mask are zeroed out at each pixel where the indices leave the image.
    The optional scale_factor will scale the final displacement.

    In-place operation.
    """
    flow_x_interp = cv2.remap(x_flow, x_indices, y_indices, cv2.INTER_NEAREST)
    flow_y_interp = cv2.remap(y_flow, x_indices, y_indices, cv2.INTER_NEAREST)
    x_mask[flow_x_interp == 0] = False
    y_mask[flow_y_interp == 0] = False

    x_indices += flow_x_interp * scale_factor
    y_indices += flow_y_interp * scale_factor
    
def flow_16bit_to_float(flow_16bit: np.ndarray):
    assert flow_16bit.dtype == np.uint16
    assert flow_16bit.ndim == 3
    h, w, c = flow_16bit.shape
    assert c == 3

    valid2D = flow_16bit[..., 2] == 1
    assert valid2D.shape == (h, w)
    assert np.all(flow_16bit[~valid2D, -1] == 0)
    valid_map = np.where(valid2D)

    # to actually compute something useful:
    flow_16bit = flow_16bit.astype('float')

    flow_map = np.zeros((h, w, 2))
    flow_map[valid_map[0], valid_map[1], 0] = (flow_16bit[valid_map[0], valid_map[1], 0] - 2 ** 15) / 128
    flow_map[valid_map[0], valid_map[1], 1] = (flow_16bit[valid_map[0], valid_map[1], 1] - 2 ** 15) / 128
    return flow_map, valid2D

def scale_optical_flow(flow, max_flow_magnitude):
    u, v = flow[0, :, :], flow[1, :, :]
    magnitude = torch.sqrt(u**2 + v**2)
    exceed_indices = magnitude > max_flow_magnitude

    u[exceed_indices] = (u[exceed_indices] / magnitude[exceed_indices]) * max_flow_magnitude
    v[exceed_indices] = (v[exceed_indices] / magnitude[exceed_indices]) * max_flow_magnitude

    return torch.stack([u, v], dim=0)

def save_flow(file_path: Path, flow: np.ndarray):
    """Save the optical flow as a 16-bit PNG."""
    height, width = flow.shape[1], flow.shape[2]
    flow_16bit = np.zeros((height, width, 3), dtype=np.uint16)
    flow_16bit[..., 1] = (flow[0] * 128 + 2**15).astype(np.uint16)  # y-component
    flow_16bit[..., 0] = (flow[1] * 128 + 2**15).astype(np.uint16)  # x-component
    imageio.v2.imwrite(str(file_path), flow_16bit, format='PNG-FI')


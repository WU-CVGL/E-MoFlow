import logging
from typing import Optional

import cv2
import numpy as np
import scipy
import torch
from torch.nn import functional

logger = logging.getLogger(__name__)

def pixel_to_normalized_coords(pixel_coords, intrinsic_matrix):
    assert pixel_coords.shape[0] == 1 and pixel_coords.shape[2] == 3, "The shape of pixel_coords should be (1, n, 3)"
    assert intrinsic_matrix.shape == (3, 3), "The shape of intrinsic_matrix should be (3, 3)"

    fx = intrinsic_matrix[0, 0]  
    fy = intrinsic_matrix[1, 1]  
    cx = intrinsic_matrix[0, 2] 
    cy = intrinsic_matrix[1, 2]

    x = pixel_coords[0, :, 0]  
    y = pixel_coords[0, :, 1]  

    x_n = (x - cx) / fx
    y_n = (y - cy) / fy

    ones = torch.ones_like(x_n)
    normalized_coords = torch.stack([x_n, y_n, ones], dim=1)  
    normalized_coords = normalized_coords.unsqueeze(0)  

    return normalized_coords

def flow_to_normalized_coords(image_plane_flow, intrinsic_matrix):
    assert image_plane_flow.shape[0] == 1 and image_plane_flow.shape[2] == 3, "The shape of pixel_coords should be (1, n, 3)"
    assert intrinsic_matrix.shape == (3, 3), "The shape of intrinsic_matrix should be (3, 3)"

    fx = intrinsic_matrix[0, 0]  
    fy = intrinsic_matrix[1, 1]  
    cx = intrinsic_matrix[0, 2]  
    cy = intrinsic_matrix[1, 2]  

    flow_x = image_plane_flow[0, :, 0]  
    flow_y = image_plane_flow[0, :, 1]  

    flow_x_n = flow_x / fx
    flow_y_n = flow_y / fy

    ones = torch.zeros_like(flow_x_n)
    normalized_plane_flow = torch.stack([flow_x_n, flow_y_n, ones], dim=1)  
    normalized_plane_flow = normalized_plane_flow.unsqueeze(0)  

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

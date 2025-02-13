import torch
import torch.nn.functional as F

def evaluate_velocity(pred_angular, pred_linear, gt_angular, gt_linear):
    # RMSE
    rmse_angular = torch.sqrt(torch.mean((pred_angular - gt_angular)**2))
    rmse_linear = torch.sqrt(torch.mean((pred_linear - gt_linear)**2))
    
    # Cosine_similarity
    cos_sim_angular = F.cosine_similarity(pred_angular, gt_angular, dim=1).mean()
    cos_sim_linear = F.cosine_similarity(pred_linear, gt_linear, dim=1).mean()
    
    return {
        "RMSE_angular": rmse_angular.item(),
        "RMSE_linear": rmse_linear.item(),
        "CosSim_angular": cos_sim_angular.item(),
        "CosSim_linear": cos_sim_linear.item()
    }

def calculate_flow_error(
    flow_gt: torch.Tensor,
    flow_pred: torch.Tensor,
    event_mask: torch.Tensor = None,
    time_scale: torch.Tensor = None,
) -> dict:
    """Calculate flow error.
    Args:
        flow_gt (torch.Tensor) ... [B x 2 x H x W]
        flow_pred (torch.Tensor) ... [B x 2 x H x W]
        event_mask (torch.Tensor) ... [B x 1 x W x H]. Optional.
        time_scale (torch.Tensor) ... [B x 1]. Optional. This will be multiplied.
            If you want to get error in 0.05 ms, time_scale should be
            `0.05 / actual_time_period`.

    Retuns:
        errors (dict) ... Key containers 'AE', 'EPE', '1/2/3PE'. all float.

    """
    # Only compute error over points that are valid in the GT (not inf or 0).
    flow_mask = torch.logical_and(
        torch.logical_and(~torch.isinf(flow_gt[:, [0], ...]), ~torch.isinf(flow_gt[:, [1], ...])),
        torch.logical_and(torch.abs(flow_gt[:, [0], ...]) > 0, torch.abs(flow_gt[:, [1], ...]) > 0),
    )  # B, H, W
    if event_mask is None:
        total_mask = flow_mask
    else:
        if len(event_mask.shape) == 3:
            event_mask = event_mask[:, None]
        total_mask = torch.logical_and(event_mask, flow_mask)
    gt_masked = flow_gt * total_mask  # b, 2, H, W
    pred_masked = flow_pred * total_mask
    n_points = torch.sum(total_mask, dim=(1, 2, 3)) + 1e-5  # B, 1

    errors = {}
    # Average endpoint error.
    if time_scale is not None:
        time_scale = time_scale.reshape(len(gt_masked), 1, 1, 1)
        gt_masked = gt_masked * time_scale
        pred_masked = pred_masked * time_scale
    endpoint_error = torch.linalg.norm(gt_masked - pred_masked, dim=1)
    errors["EPE"] = torch.mean(torch.sum(endpoint_error, dim=(1, 2)) / n_points)
    errors["1PE"] = torch.mean(torch.sum(endpoint_error > 1, dim=(1, 2)) / n_points)
    errors["2PE"] = torch.mean(torch.sum(endpoint_error > 2, dim=(1, 2)) / n_points)
    errors["3PE"] = torch.mean(torch.sum(endpoint_error > 3, dim=(1, 2)) / n_points)

    # Angular error
    u, v = pred_masked[:, 0, ...], pred_masked[:, 1, ...]
    u_gt, v_gt = gt_masked[:, 0, ...], gt_masked[:, 1, ...]
    cosine_similarity = (1.0 + u * u_gt + v * v_gt) / (torch.sqrt(1 + u * u + v * v) * torch.sqrt(1 + u_gt * u_gt + v_gt * v_gt))
    cosine_similarity = torch.clamp(cosine_similarity, -1, 1)
    errors["AE"] = torch.mean(torch.sum(torch.acos(cosine_similarity), dim=(1, 2)) / n_points)
    errors["AE"] = errors["AE"] * (180.0 / torch.pi)
    return errors

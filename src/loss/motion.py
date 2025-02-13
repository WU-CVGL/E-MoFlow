import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionLoss(nn.Module):
    def __init__(
        self, 
        alpha=0.5, 
        beta=0.5,
        delta_ang_mag=0.5,  
        delta_lin_mag=1.0,  
        delta_dir=0.2
    ):     
        """
        Args:
            alpha (float): Weight for angular/linear velocity magnitude loss
            beta (float): Weight for directional consistency loss
            delta_ang_mag (float): Huber threshold for angular velocity magnitude (unit: rad/s)
            delta_lin_mag (float): Huber threshold for linear velocity magnitude (unit: m/s)
            delta_dir (float): Huber threshold for directional discrepancy (unit: cosine distance)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.huber_ang_mag = nn.HuberLoss(delta=delta_ang_mag)
        self.huber_lin_mag = nn.HuberLoss(delta=delta_lin_mag)
        self.huber_dir = nn.HuberLoss(delta=delta_dir)

    def forward(self, pred_angular, pred_linear, gt_angular, gt_linear):
        ang_mag_diff = torch.norm(pred_angular, dim=1) - torch.norm(gt_angular, dim=1)
        lin_mag_diff = torch.norm(pred_linear, dim=1) - torch.norm(gt_linear, dim=1)
        
        loss_angular_mag = self.huber_ang_mag(ang_mag_diff, torch.zeros_like(ang_mag_diff))
        loss_linear_mag = self.huber_lin_mag(lin_mag_diff, torch.zeros_like(lin_mag_diff))

        cos_sim_angular = F.cosine_similarity(pred_angular, gt_angular, dim=1)
        cos_sim_linear = F.cosine_similarity(pred_linear, gt_linear, dim=1)
        dir_diff_angular = 1 - cos_sim_angular  # 0~2
        dir_diff_linear = 1 - cos_sim_linear
        
        loss_angular_dir = self.huber_dir(dir_diff_angular, torch.zeros_like(dir_diff_angular))
        loss_linear_dir = self.huber_dir(dir_diff_linear, torch.zeros_like(dir_diff_linear))

        motion_loss = (
            self.alpha * (loss_angular_mag + loss_linear_mag) +
            self.beta * (loss_angular_dir + loss_linear_dir)
        )
        
        return motion_loss

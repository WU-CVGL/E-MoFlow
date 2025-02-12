import torch
import torch.nn as nn
import torch.nn.functional as F

class VelocityLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.6):

        """
        Args:
            alpha: Weight for angular velocity MSE loss
            beta: Weight for linear velocity MSE loss
            gamma: Weight for directional consistency loss (applied to both angular and linear velocity)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss()
    
    def forward(self, pred_angular, pred_linear, gt_angular, gt_linear):
        # value loss
        loss_angular_mse = self.mse(pred_angular, gt_angular)
        loss_linear_mse = self.mse(pred_linear, gt_linear)
        
        # dir loss
        cos_sim_angular = F.cosine_similarity(pred_angular, gt_angular, dim=1)
        cos_sim_linear = F.cosine_similarity(pred_linear, gt_linear, dim=1)
        loss_angular_dir = (1 - cos_sim_angular).mean()  
        loss_linear_dir = (1 - cos_sim_linear).mean()
        
        # motion loss
        total_loss = (
            self.alpha * loss_angular_mse +
            self.beta * loss_linear_mse +
            self.gamma * (loss_angular_dir + loss_linear_dir)
        )
        return total_loss

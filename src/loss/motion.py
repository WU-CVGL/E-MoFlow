import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionLoss(nn.Module):
    def __init__(
        self, 
        alpha: float = 0.5, 
        beta: float = 0.5,
        loss_type: str = "MSE"
    ):     
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss(delta=1.0)
        self.loss_type = loss_type

    def forward(self, pred_angular, pred_linear, gt_angular, gt_linear):
        if self.loss_type == "MSE":
            loss_linear = self.mse_loss(pred_linear, gt_linear)
            loss_angular = self.mse_loss(pred_angular, gt_angular)
        elif self.loss_type == "Huber":
            loss_linear = self.huber_loss(pred_linear, gt_linear)
            loss_angular = self.huber_loss(pred_angular, gt_angular)
        else:
            raise TypeError("Unknown loss type")
        motion_loss = self.alpha * loss_linear + self.beta * loss_angular
        
        return motion_loss
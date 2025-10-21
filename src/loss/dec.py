import torch
import torch.nn as nn

from src.utils.vector_math import vec2skewmat

class DifferentialEpipolarLoss(nn.Module):
    def __init__(self, delta=0.1, use_huber=True):
        super(DifferentialEpipolarLoss, self).__init__()
        self.delta = delta  
        self.use_huber = use_huber  

    def huber_loss(self, error):
        """
        Compute Huber loss
        Args:
            error: Error tensor
        Returns:
            Huber loss
        """
        abs_error = torch.abs(error)
        quadratic = torch.minimum(abs_error, torch.tensor(self.delta))
        linear = abs_error - quadratic
        return 0.5 * quadratic**2 + self.delta * linear
    
    def forward(self, coord, flow, lin_vel, ang_vel):
        """
        Args:
            coord:   Input coordinates tensor of shape (B, N, 3) or (B, 3, N)
            flow:    Optical flow tensor of shape (B, N, 3) or (B, 3, N)
            lin_vel: Linear velocity tensor of shape (B, 3)
            ang_vel: Angular velocity tensor of shape (B, 3)

        Returns:
            dec_loss: Per-point loss tensor of shape (B, N) or (1, N) for backward compatibility
            average_dec_loss: Scalar mean loss
        """
        # (B, 3, N)
        if coord.shape[-1] == 3:
            x_batch_tensor = coord.transpose(1, 2)  
            u_batch_tensor = flow.transpose(1, 2) 
        else:
            # Format already (B, 3, N)
            x_batch_tensor = coord
            u_batch_tensor = flow

        B = x_batch_tensor.shape[0]

        # batch dec loss
        v_skew = vec2skewmat(lin_vel)  # (B, 3, 3)
        w_skew = vec2skewmat(ang_vel)  # (B, 3, 3)
        s = 0.5 * (torch.bmm(v_skew, w_skew) + torch.bmm(w_skew, v_skew))  # (B, 3, 3)
        term1 = torch.einsum("bin,bij,bjn->bn", u_batch_tensor, v_skew, x_batch_tensor)  # [B, N]
        term2 = torch.einsum("bin,bij,bjn->bn", x_batch_tensor, s, x_batch_tensor)      # [B, N]
        error = term1 - term2  # [B, N]

        if self.use_huber:
            dec_loss = self.huber_loss(error)
        else:
            dec_loss = torch.square(error)

        average_dec_loss = torch.mean(dec_loss)

        return dec_loss, average_dec_loss

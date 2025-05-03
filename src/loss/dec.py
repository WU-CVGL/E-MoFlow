import torch
import torch.nn as nn

from src.utils.vector_math import vector_to_skew_matrix

class DifferentialEpipolarLoss(nn.Module):
    def __init__(self):
        super(DifferentialEpipolarLoss, self).__init__()

    def forward(self, coord, flow, lin_vel, ang_vel):
        """
        Args:
            coord:   Input coordinates tensor of shape (B, 3, N)
            flow:    Optical flow tensor of shape (B, 3, N)
            lin_vel: Linear velocity tensor of shape (B, 3)
            ang_vel: Angular velocity tensor of shape (B, 3)
        
        Returns:
            dec_loss: Per-point loss tensor of shape (1, N)
            average_dec_loss: Scalar mean loss
        """
        # Convert velocities to skew-symmetric matrices
        v_skew = vector_to_skew_matrix(lin_vel).squeeze(0)  # (3, 3)
        w_skew = vector_to_skew_matrix(ang_vel).squeeze(0)  # (3, 3)
        x_batch_tensor = coord.squeeze(0).transpose(0,1) # [3,N]
        u_batch_tensor = flow.squeeze(0).transpose(0,1) # [3,N]
        
        # Compute symmetric matrix S
        s = 0.5 * (torch.mm(v_skew, w_skew) + torch.mm(w_skew, v_skew)) # (3, 3)

        # Einstein summation terms
        term1 = torch.einsum("in,ij,jn->n", u_batch_tensor, v_skew, x_batch_tensor)
        term2 = torch.einsum("in,ij,jn->n", x_batch_tensor, s, x_batch_tensor)

        error = term1 - term2
        dec_loss = torch.square(error)
        average_dec_loss = torch.mean(dec_loss)

        return dec_loss, average_dec_loss

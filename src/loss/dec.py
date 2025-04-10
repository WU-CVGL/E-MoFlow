import torch
from src.utils.vector_math import vector_to_skew_matrix

def differential_epipolar_constrain(coord, flow ,lin_vel, ang_vel):
    x_batch_tensor = coord.squeeze(0)
    u_batch_tensor = flow.squeeze(0)
    v_skew = vector_to_skew_matrix(lin_vel).squeeze(0)
    w_skew = vector_to_skew_matrix(ang_vel).squeeze(0)
    
    s = 0.5 * (torch.mm(v_skew, w_skew) + torch.mm(w_skew, v_skew))
    
    v_skew_x = torch.matmul(v_skew, x_batch_tensor.transpose(0,1)).transpose(0,1) # [N,3]
    s_x = torch.matmul(s, x_batch_tensor.transpose(0,1)).transpose(0,1)  # [N,3]
    
    term1 = torch.sum(u_batch_tensor * v_skew_x, dim=1) # [N]
    term2 = torch.sum(x_batch_tensor * s_x, dim=1)  # [N]
    error = term1 - term2   
    error = error.unsqueeze(0)
    dec_loss = torch.square(error)
    average_dec_loss = torch.mean(dec_loss)
    
    return dec_loss, average_dec_loss
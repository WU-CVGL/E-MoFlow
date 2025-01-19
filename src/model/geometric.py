import torch
import kornia
import theseus as th

from typing import Optional, Tuple
from src.utils.vector_math import vector_to_skew_matrix

class Pixel2Cam:
    def __init__(self, H: int, W: int, device: torch.device):
        self.grid = kornia.utils.create_meshgrid(
            H, W, normalized_coordinates=False).to(device)
        self.grid = self.grid.squeeze(0)
        self.ones = torch.ones(H, W, 1, device=device)
        self.pixels_homogeneous = torch.cat(
            [self.grid, self.ones], dim=-1)  # [H,W,3]
        
    def __call__(self, K: torch.Tensor, depth: Optional[torch.Tensor] = None) -> torch.Tensor:
        if K.dim() == 2:
            K = K.unsqueeze(0)
        B = K.shape[0]
        
        pixels_batch = self.pixels_homogeneous.unsqueeze(0).expand(B, -1, -1, -1).to(K.device)
        K_4x4 = torch.zeros(B, 4, 4, device=K.device)
        K_4x4[:, :3, :3] = K
        K_4x4[:, 3, 3] = 1.0
        K_4x4_inv = torch.inverse(K_4x4)
        
        if depth is None:
            depth = torch.ones(B, 1, *self.grid.shape[:2], device=K.device)
            
        return kornia.geometry.camera.pinhole.pixel2cam(
            depth, K_4x4_inv, pixels_batch)[..., :3]

class PoseOptimizer:
    def __init__(self, image_size, device="cuda"):
        self.H, self.W = image_size
        self.device = device
    
    def compute_B_matrix(self, x):
        # x shape: [N,3] normalized coordinate
        xx = x[:, 0]  # x coordinate
        yy = x[:, 1]  # y coordinate
        
        B_1 = torch.stack([xx*yy, -(1+xx*xx), yy], dim=1)  # [N,3]
        B_2 = torch.stack([1+yy*yy, -xx*yy, -xx], dim=1)   # [N,3]
        
        return torch.stack([B_1, B_2], dim=1)  # [N,2,3]
    
    def dec_error_fn(self, optim_vars, aux_vars):
        v_vec, w_vec = optim_vars
        x_batch, u_batch = aux_vars
        v_vec_tensor, w_vec_tensor = v_vec.tensor, w_vec.tensor
        x_batch_tensor, u_batch_tensor = x_batch.tensor, u_batch.tensor
        
        x_batch_tensor = x_batch_tensor.squeeze(0)
        u_batch_tensor = u_batch_tensor.squeeze(0)
        v_skew = vector_to_skew_matrix(v_vec_tensor).squeeze(0)
        w_skew = vector_to_skew_matrix(w_vec_tensor).squeeze(0)
        
        s = 0.5 * (torch.mm(v_skew, w_skew) + torch.mm(w_skew, v_skew))
        
        v_skew_x = torch.matmul(v_skew, x_batch_tensor.transpose(0,1)).transpose(0,1) # [N,3]
        s_x = torch.matmul(s, x_batch_tensor.transpose(0,1)).transpose(0,1)  # [N,3]
        
        term1 = torch.sum(u_batch_tensor * v_skew_x, dim=1) # [N]
        term2 = torch.sum(x_batch_tensor * s_x, dim=1)  # [N]
        error = term1 - term2
        
        return error.unsqueeze(0)    
    
    def rotation_error_fn(self, optim_vars, aux_vars):
        w_vec = optim_vars[0]
        x_batch, u_batch = aux_vars
        w_vec_tensor = w_vec.tensor
        x_batch_tensor = x_batch.tensor.squeeze(0)  # [N,3]
        u_batch_tensor = u_batch.tensor.squeeze(0)  # [N,3]
        
        B = self.compute_B_matrix(x_batch_tensor)  # [N,2,3]
        est_flow = (B @ w_vec_tensor.t()).squeeze(-1)

        error = u_batch_tensor[..., :2] - est_flow  # [N,2]
        return error.reshape(-1).unsqueeze(0)  
    
    def velocity_constraint_fn(self, optim_vars, aux_vars=None):
        v_vec = optim_vars[0]
        v_norm = torch.norm(v_vec.tensor, p=2, dim=-1)
        return (v_norm - 1.0).unsqueeze(0)  # [1,1] shape
    
    def create_cost_function(self, normalized_coords, optical_flow, only_rotation=False):
        x = th.Variable(normalized_coords, name="norm_coords")  # [1,N,3]
        u = th.Variable(optical_flow, name="optical_flow")      # [1,N,3]
        
        if not only_rotation:     
            v = th.Vector(dof=3, name="linear_velocity") 
            w = th.Vector(dof=3, name="angular_velocity") 
            optim_vars = [v, w]
            aux_vars = [x, u]
            w_dec_value = torch.tensor(1.0 / x.shape[1], dtype=torch.float32)
            w_dec = th.ScaleCostWeight(torch.sqrt(w_dec_value))
            w_norm_value = torch.tensor(100, dtype=torch.float32) 
            w_norm = th.ScaleCostWeight(torch.sqrt(w_norm_value))
        
            dec_cost_fn = th.AutoDiffCostFunction(
                optim_vars,
                self.dec_error_fn,
                x.shape[1],
                w_dec,
                aux_vars=aux_vars,
                name="dec_cost_fn"
            )
        
            vel_cost_fn = th.AutoDiffCostFunction(
                [v],
                self.velocity_constraint_fn,
                1,
                w_norm,
                aux_vars=None,
                name="velocity_constraint"
            )
        
            return [dec_cost_fn, vel_cost_fn]
        
        else:
            w = th.Vector(dof=3, name="angular_velocity")
            optim_vars = [w]
            aux_vars = [x, u]
            w_rot_value = torch.tensor(1.0 / (x.shape[1] * 2), dtype=torch.float32)
            w_rot = th.ScaleCostWeight(torch.sqrt(w_rot_value))
            
            rot_cost_fn = th.AutoDiffCostFunction(
                optim_vars,
                self.rotation_error_fn,
                x.shape[1] * 2,
                w_rot,
                aux_vars=aux_vars,
                name="rot_cost_fn"
            )
            
            return [rot_cost_fn]
    
    def optimize(self, normalized_coords, optical_flow, only_rotation=False, init_velc=None, num_iterations=5000):
        # check
        assert normalized_coords.shape[-1] == 3, "[ERROR] Normalized coordinates should be in homogeneous form [x,y,1]"
        assert optical_flow.shape[-1] == 3, "[ERROR] Optical flow should be in homogeneous form [u,v,0]"
        assert normalized_coords.shape[:-1] == optical_flow.shape[:-1], "[ERROR] Batch dimensions should match"
        
        normalized_coords = normalized_coords.unsqueeze(0)
        optical_flow = optical_flow.unsqueeze(0)
        
        objective = th.Objective()
        cost_functions = self.create_cost_function(
            normalized_coords, optical_flow, only_rotation
        )
        for cost_fn in cost_functions:
            objective.add(cost_fn)
            
        optimizer = th.LevenbergMarquardt(
            objective,
            max_iterations=num_iterations,
            step_size=1,
        )
        
        if init_velc is not None:
            if only_rotation:
                w_init = init_velc[1] if len(init_velc) > 1 else init_velc[0]
            else:
                v_init, w_init = init_velc
        else:
            if not only_rotation:
                v_init = torch.rand((1,3), device=self.device) 
                v_init = v_init / torch.norm(v_init+1e-9, p=2, dim=-1, keepdim=True)
            w_init = torch.rand((1,3), device=self.device) 
        
        theseus_inputs = {
            "norm_coords": normalized_coords,
            "optical_flow": optical_flow
        }
        if not only_rotation:
            theseus_inputs["linear_velocity"] = v_init
        theseus_inputs["angular_velocity"] = w_init
        
        theseus_optim = th.TheseusLayer(optimizer).to(self.device)
        with torch.no_grad():
            updated_inputs, info = theseus_optim.forward(
                theseus_inputs,
                optimizer_kwargs={
                    # "damping": 0.1,
                    "track_best_solution": True,
                    "track_state_history": True,
                    "track_err_history": True,
                    "verbose": True,
                },
            )
        
        if only_rotation:
            w_opt = info.best_solution["angular_velocity"]
            w_history = info.state_history["angular_velocity"].view(-1, 3)
            err = info.err_history.view(-1,1)
            return None, w_opt, None, w_history, err
        else:
            v_opt = info.best_solution["linear_velocity"] 
            w_opt = info.best_solution["angular_velocity"]
            v_history = info.state_history["linear_velocity"].view(-1, 3)
            w_history = info.state_history["angular_velocity"].view(-1, 3)
            err = info.err_history.view(-1,1)
            return v_opt, w_opt, v_history, w_history, err
        
def compute_motion_field(coords, v_gt, w_gt, depth_gt):
    """
    Compute motion field equation: u = Av/Z + Bw
    
    Args:
        coords: Normalized coordinates, shape (H,W,3) each point is (x,y,1)
        v_gt: Linear velocity, shape (1,3)
        w_gt: Angular velocity, shape (1,3)
        depth_gt: Depth map, shape (H,W,1)
    
    Returns:
        optical_flow: Optical flow field, shape (H,W,3), each point is (u,v,0)
    """
    
    x = coords[..., 0].unsqueeze(-1)  # (H,W,1)
    y = coords[..., 1].unsqueeze(-1)  # (H,W,1)
    
    A = torch.zeros((*coords.shape[:2], 2, 3), device=coords.device)
    A[..., 0, 0] = -1
    A[..., 0, 2] = x.squeeze(-1)
    A[..., 1, 1] = -1  
    A[..., 1, 2] = y.squeeze(-1)
    
    B = torch.zeros_like(A)
    B[..., 0, 0] = x.squeeze(-1) * y.squeeze(-1)
    B[..., 0, 1] = -(1 + x.squeeze(-1)**2)
    B[..., 0, 2] = y.squeeze(-1)
    B[..., 1, 0] = 1 + y.squeeze(-1)**2
    B[..., 1, 1] = -x.squeeze(-1) * y.squeeze(-1)
    B[..., 1, 2] = -x.squeeze(-1)
    
    v_gt_expanded = v_gt.expand(coords.shape[0], coords.shape[1], -1)
    
    Av = torch.matmul(A, v_gt_expanded.unsqueeze(-1)).squeeze(-1)
    Av_Z = Av / depth_gt 
    
    w_gt_expanded = w_gt.expand(coords.shape[0], coords.shape[1], -1)
    
    Bw = torch.matmul(B, w_gt_expanded.unsqueeze(-1)).squeeze(-1)
    
    flow_2d = Av_Z + Bw  # (H,W,2)
    
    optical_flow = torch.zeros_like(coords)
    optical_flow[..., :2] = flow_2d
    
    return optical_flow
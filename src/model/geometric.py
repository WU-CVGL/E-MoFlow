import torch
import kornia
import theseus as th
import torch.nn as nn

from typing import Optional, Tuple, cast
from src.utils.vector_math import vector_to_skew_matrix

class Pixel2Cam:
    def __init__(self, H: int, W: int, K: torch.Tensor, device: torch.device):
        self.H = H
        self.W = W
        self.K = K.to(device)
        self.device = device
        self.grid = kornia.utils.create_meshgrid(
            H, W, normalized_coordinates=False).to(device)
        self.grid = self.grid.squeeze(0)
        self.ones = torch.ones(H, W, 1, device=device)
        self.pixels_homogeneous = torch.cat(
            [self.grid, self.ones], dim=-1)  # [H,W,3]
        
    def generate_image_coordinate(
        self,
    ) -> torch.Tensor:
        if self.K.dim() == 2:
            self.K = self.K.unsqueeze(0)
        B = self.K.shape[0]
        
        pixels_batch = self.pixels_homogeneous.unsqueeze(0).expand(B, -1, -1, -1).to(self.device)

        return pixels_batch
        
    def generate_normalized_image_coordinate(
        self, 
        # K: torch.Tensor, 
        depth: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.K.dim() == 2:
            self.K = self.K.unsqueeze(0)
        B = self.K.shape[0]
        
        pixels_batch = self.pixels_homogeneous.unsqueeze(0).expand(B, -1, -1, -1).to(self.device)
        K_4x4 = torch.zeros(B, 4, 4, device=self.K.device)
        K_4x4[:, :3, :3] = self.K
        K_4x4[:, 3, 3] = 1.0
        K_4x4_inv = torch.inverse(K_4x4)
        
        if depth is None:
            depth = torch.ones(B, 1, self.H, self.W, device=self.K.device)
            
        return kornia.geometry.camera.pinhole.pixel2cam(
            depth, K_4x4_inv, pixels_batch)[..., :3]

    def sample_sparse_points(
        self, 
        sparsity_level: int,
        norm_coords: torch.Tensor, 
    ) -> torch.Tensor:
        B, H, W, _ = norm_coords.shape
        h_split, w_split = self._get_splits(sparsity_level)
        
        if H % h_split != 0 or W % w_split != 0:
            raise ValueError(
                f"Cannot divide {H}x{W} image into {h_split}x{w_split} blocks. "
                f"Please ensure height {H} is divisible by {h_split} and width {W} is divisible by {w_split}."
            )

        block_h = H // h_split
        block_w = W // w_split

        base_rows = torch.arange(h_split, device=norm_coords.device).view(1, h_split, 1) * block_h
        base_cols = torch.arange(w_split, device=norm_coords.device).view(1, 1, w_split) * block_w
        row_offsets = torch.randint(0, block_h, (B, h_split, w_split), device=norm_coords.device)
        col_offsets = torch.randint(0, block_w, (B, h_split, w_split), device=norm_coords.device)
        row_indices = base_rows + row_offsets
        col_indices = base_cols + col_offsets

        b_idx = torch.arange(B, device=norm_coords.device)[:, None, None]

        sampled_points = norm_coords[b_idx, row_indices, col_indices]

        return sampled_points.reshape(B, -1, 3)

    def _get_splits(self, n: int) -> Tuple[int, int]:
        max_factor = int(n**0.5)
        for i in range(max_factor, 0, -1):
            if n % i == 0:
                return i, n // i
        return 1, n
    
    def sample_sparse_coordinates(
        self,
        coord_tensor: torch.Tensor, 
        mask: torch.Tensor, 
        n: int
    ) -> torch.Tensor:
        assert coord_tensor.dim() == 4 and coord_tensor.shape[0] == 1, "The coordinate tensor shape should be [1, H, W, 3]."
        H, W = coord_tensor.shape[1], coord_tensor.shape[2]
        
        if mask == None:
            valid_coords = coord_tensor[0].view(-1, 3)
        else:
            assert mask.shape == (H, W), "The mask shape does not match the coordinate tensor."
            valid_coords = coord_tensor[0][mask.bool()]  # [num_valid, 3]
        num_valid = valid_coords.shape[0]
        
        if num_valid == 0:
            return torch.zeros((1, 0, 3), device=coord_tensor.device)
        
        n_samples = min(n, num_valid)
    
        rand_idx = torch.randperm(num_valid, device=valid_coords.device)[:n_samples]
        sampled_coords = valid_coords[rand_idx]  # [n_samples, 3]
        
        # [1, n_samples, 3]
        return sampled_coords.unsqueeze(0)

from enum import Enum

class MotionModel(Enum):
    PURE_TRANSLATION = "pure_translation"
    PURE_ROTATION = "pure_rotation"
    SIXDOF_MOTION = "6dof_motion"

class DiffEpipolarTheseusOptimizer:
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
        
        x_batch_tensor = x_batch_tensor.squeeze(0).transpose(0,1) # [3,N]
        u_batch_tensor = u_batch_tensor.squeeze(0).transpose(0,1) # [3,N]
        v_skew = vector_to_skew_matrix(v_vec_tensor).squeeze(0)   # [3,3]
        w_skew = vector_to_skew_matrix(w_vec_tensor).squeeze(0)   # [3,3]
        
        s = 0.5 * (torch.mm(v_skew, w_skew) + torch.mm(w_skew, v_skew))
        
        term1 = torch.einsum("in,ij,jn->n", u_batch_tensor, v_skew, x_batch_tensor)
        term2 = torch.einsum("in,ij,jn->n", x_batch_tensor, s, x_batch_tensor)
        error = term1 - term2
        error = error.unsqueeze(0)
        
        return error
    
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
    
    def translation_error_fn(self, optim_vars, aux_vars):
        v_vec = optim_vars[0]
        x_batch, u_batch = aux_vars
        v_vec_tensor = v_vec.tensor
        x_batch_tensor, u_batch_tensor = x_batch.tensor, u_batch.tensor
        
        x_batch_tensor = x_batch_tensor.squeeze(0).transpose(0,1)  # [3,N]
        u_batch_tensor = u_batch_tensor.squeeze(0).transpose(0,1)  # [3,N]
        v_skew = vector_to_skew_matrix(v_vec_tensor).squeeze(0)    # [3,3]
        
        term1 = torch.einsum("in,ij,jn->n", u_batch_tensor, v_skew, x_batch_tensor)  # shape (N,)
        error = term1.unsqueeze(0)
        
        return error
    
    def velocity_constraint_fn(self, optim_vars, aux_vars=None):
        v_vec = optim_vars[0]
        v_norm = torch.norm(v_vec.tensor, p=2, dim=-1)
        return 1000 * (v_norm - 1.0).unsqueeze(0)  # [1,1] shape
    
    def create_cost_function(self, normalized_coords, optical_flow, motion_model: MotionModel):
        x = th.Variable(normalized_coords, name="norm_coords")  # [1,N,3]
        u = th.Variable(optical_flow, name="optical_flow")      # [1,N,3]
        
        if motion_model == MotionModel.SIXDOF_MOTION:     
            v = th.Vector(
                tensor=torch.zeros(1, 3, dtype=torch.float32, device=self.device), 
                name="linear_velocity"
            ) 
            w = th.Vector(
                tensor=torch.zeros(1, 3, dtype=torch.float32, device=self.device), 
                name="angular_velocity"
            ) 
            optim_vars = [v, w]
            aux_vars = [x, u]
            # w_dec_value = torch.tensor((1 / x.shape[1]), dtype=torch.float32)
            # w_dec = th.ScaleCostWeight(torch.sqrt(w_dec_value))
            # w_norm_value = torch.tensor(1, dtype=torch.float32) 
            # w_norm = th.ScaleCostWeight(torch.sqrt(w_norm_value))

            dec_cost_fn = th.AutoDiffCostFunction(
                optim_vars,
                self.dec_error_fn,
                x.shape[1],
                # w_dec,
                aux_vars=aux_vars,
                name="dec_cost_fn"
            )
            
            # log_loss_radius = th.Vector(1, name="log_loss_radius", dtype=torch.float32)
            # robust_dec_cost_fn = th.RobustCostFunction(
            #     dec_cost_fn,
            #     th.HuberLoss,
            #     log_loss_radius,
            #     name=f"robust_{dec_cost_fn.name}",
            # )
        
            vel_cost_fn = th.AutoDiffCostFunction(
                [v],
                self.velocity_constraint_fn,
                1,
                # w_norm,
                aux_vars=None,
                name="velocity_constraint"
            )
        
            return [dec_cost_fn, vel_cost_fn]
        
        elif motion_model == MotionModel.PURE_ROTATION:
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
        
        elif motion_model == MotionModel.PURE_TRANSLATION:
            v = th.Vector(
                tensor=torch.zeros(1, 1, 3, dtype=torch.float32, device=self.device), 
                name="linear_velocity"
            ) 
            optim_vars = [v]
            aux_vars = [x, u]
            # w_dec_value = torch.tensor((1 / x.shape[1]), dtype=torch.float32, device=self.device)
            # w_dec = th.ScaleCostWeight(torch.sqrt(w_dec_value))
            # w_norm_value = torch.tensor(1, dtype=torch.float32, device=self.device) 
            # w_norm = th.ScaleCostWeight(torch.sqrt(w_norm_value))

            trans_cost_fn = th.AutoDiffCostFunction(
                optim_vars,
                self.translation_error_fn,
                x.shape[1],
                # w_dec,
                aux_vars=aux_vars,
                name="trans_cost_fn"
            )
            
            log_loss_radius = th.Vector(
                tensor=torch.zeros(1, 1, dtype=torch.float32, device=self.device),
                name="log_loss_radius",
            )
            robust_trans_cost_fn = th.RobustCostFunction(
                trans_cost_fn,
                th.HuberLoss,
                log_loss_radius,
                name=f"robust_{trans_cost_fn.name}",
            )
        
            vel_cost_fn = th.AutoDiffCostFunction(
                [v],
                self.velocity_constraint_fn,
                1,
                # w_norm,
                aux_vars=None,
                name="velocity_constraint"
            )
        
            return [robust_trans_cost_fn, vel_cost_fn]
        
        else:
            raise ValueError("Unknow motion model")
            
    def optimize(self, normalized_coords, optical_flow, motion_model: MotionModel, init_velc=None, num_iterations=5000):
        # check
        assert normalized_coords.shape[-1] == 3, "[ERROR] Normalized coordinates should be in homogeneous form [x,y,1]"
        assert optical_flow.shape[-1] == 3, "[ERROR] Optical flow should be in homogeneous form [u,v,0]"
        assert normalized_coords.shape[:-1] == optical_flow.shape[:-1], "[ERROR] Batch dimensions should match"
        
        if normalized_coords.dim() == 2:
            normalized_coords = normalized_coords.unsqueeze(0)
        if optical_flow.dim() == 2:
            optical_flow = optical_flow.unsqueeze(0)       
              
        objective = th.Objective()
        cost_functions = self.create_cost_function(
            normalized_coords, optical_flow, motion_model
        )
        for cost_fn in cost_functions:
            objective.add(cost_fn)
            
        optimizer = th.LevenbergMarquardt(
            objective,
            linear_solver_cls=th.CholeskyDenseSolver,
            linearization_cls=th.DenseLinearization,
            vectorize=True,
            empty_cuda_cache=True,
            step_size=0.1,
            abs_err_tolerance=1.0e-20,
            rel_err_tolerance=1.0e-8,
            kwargs={
                "backward_mode": th.BackwardMode.UNROLL,
                "max_iterations": num_iterations,
                "damping": 1.0,
                "adaptive_damping": True,
                "verbose": False,
                "track_err_history": True,
                "track_state_history": True,
            }
        )

        if init_velc is not None:
            if motion_model == MotionModel.PURE_TRANSLATION:
                v_init = init_velc[0]
            elif motion_model == MotionModel.PURE_ROTATION:
                w_init = init_velc[1] if len(init_velc) > 1 else init_velc[0]
            elif motion_model == MotionModel.SIXDOF_MOTION:
                v_init, w_init = init_velc
            else:
                raise ValueError("Unknow motion model")   
        else:
            if motion_model == MotionModel.PURE_TRANSLATION:
                v_init = torch.randn((1,3), device=self.device) 
                v_init = v_init / torch.norm(v_init+1e-9, p=2, dim=-1, keepdim=True)
            elif motion_model == MotionModel.PURE_ROTATION:
                w_init = torch.randn((1,3), device=self.device)
            elif motion_model == MotionModel.SIXDOF_MOTION:
                v_init = torch.randn((1,3), device=self.device) 
                v_init = v_init / torch.norm(v_init+1e-9, p=2, dim=-1, keepdim=True)
                w_init = torch.randn((1,3), device=self.device)
            else:
                raise ValueError("Unknow motion model")
            
        theseus_inputs = {
            "norm_coords": normalized_coords,
            "optical_flow": optical_flow,
            # "log_loss_radius": torch.log(torch.tensor([5e-3])).unsqueeze(1).to(self.device)
        }
        if motion_model == MotionModel.PURE_TRANSLATION:
            theseus_inputs["linear_velocity"] = v_init
        elif motion_model == MotionModel.PURE_ROTATION:
            theseus_inputs["angular_velocity"] = w_init
        elif motion_model == MotionModel.SIXDOF_MOTION:
            theseus_inputs["linear_velocity"] = v_init
            theseus_inputs["angular_velocity"] = w_init
        else:
            raise ValueError("Unknow motion model")
        
        theseus_layer = th.TheseusLayer(optimizer).to(self.device)
        updated_inputs, info = theseus_layer.forward(
            theseus_inputs,
            optimizer_kwargs={
                "backward_mode": th.BackwardMode.UNROLL,
                "max_iterations": num_iterations,
                "damping": 1.0, 
                "adaptive_damping": True,
                "verbose": False,
                "track_err_history": True,
                "track_state_history": True,
            }
        )
        
        final_linearization = optimizer.linear_solver.linearization
        hessian = final_linearization.AtA  # [6,6]
        
        if motion_model == MotionModel.PURE_ROTATION:
            w_opt = updated_inputs["angular_velocity"]
            w_history = info.state_history["angular_velocity"].view(-1, 3)
            err = info.err_history.view(-1,1)
            return w_opt, w_history, err
        elif motion_model == MotionModel.PURE_TRANSLATION:
            v_opt = updated_inputs["linear_velocity"]
            v_history = info.state_history["linear_velocity"].view(-1, 3)
            err = info.err_history.view(-1,1)
            return v_opt, v_history, err
        elif motion_model == MotionModel.SIXDOF_MOTION:
            v_opt = updated_inputs["linear_velocity"]
            w_opt = updated_inputs["angular_velocity"]
            v_history = info.state_history["linear_velocity"].view(-1, 3)
            w_history = info.state_history["angular_velocity"].view(-1, 3)
            err = info.err_history.view(-1,1)
            return v_opt, w_opt, v_history, w_history, err
        
def compute_motion_field(K, coords, v_gt, w_gt, depth_gt):
    """
    Compute motion field equation: u = Av/Z + Bw
    
    Args:
        K: intrinsic mat
        coords: Normalized coordinates, shape (H,W,3) each point is (x,y,1)
        v_gt: Linear velocity, shape (1,3)
        w_gt: Angular velocity, shape (1,3)
        depth_gt: Depth map, shape (H,W,1)
    
    Returns:
        norm_optical_flow: Optical flow field on normalized plane, shape (H,W,3), each point is (u,v,0)
        optical_flow: Optical flow field on pixel plane, shape (H,W,3), each point is (u,v,0)
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
    
    norm_optical_flow = torch.zeros_like(coords)
    norm_optical_flow[..., :2] = flow_2d
    
    fx, fy = K[0,0], K[1,1]
    optical_flow = norm_optical_flow.clone()
    optical_flow[..., 0] *= fx
    optical_flow[..., 1] *= fy
    
    return norm_optical_flow, optical_flow
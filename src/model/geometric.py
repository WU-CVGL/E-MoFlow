import math
import torch
import kornia
import theseus as th

from typing import Any, List, Optional, Tuple, Union

def vector_angle(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    v1_norm = torch.norm(v1, dim=-1)
    v2_norm = torch.norm(v2, dim=-1)
    
    zero_mask = (v1_norm < eps) | (v2_norm < eps)
    
    dot_product = torch.sum(v1 * v2, dim=-1)
    cosine = dot_product / (v1_norm * v2_norm + eps)
    cosine = torch.clamp(cosine, -1.0 + eps, 1.0 - eps)
    angle = torch.acos(cosine)
    
    angle = torch.where(zero_mask, torch.zeros_like(angle), angle)
    return angle

def vector_angle_degree(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    angle_rad = vector_angle(v1, v2, eps)
    return angle_rad * 180 / torch.pi

def quat2angvel(q1: torch.Tensor, q2: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Convert quaternion difference to angular velocity
    
    Args:
        q1: First quaternion [w,x,y,z]
        q2: Second quaternion [w,x,y,z]
        dt: Time difference
    
    Returns:
        omega: Angular velocity [3]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    omega = 2/dt * torch.tensor([
        w1*x2 - x1*w2 - y1*z2 + z1*y2,
        w1*y2 + x1*z2 - y1*w2 - z1*x2,
        w1*z2 - x1*y2 + y1*x2 - z1*w2
    ])
    
    return omega

def slerp(q1: torch.Tensor, q2: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical Linear Interpolation between quaternions"""
    cos_half_theta = torch.dot(q1, q2)
    
    if cos_half_theta < 0:
        q2 = -q2
        cos_half_theta = -cos_half_theta
    
    if cos_half_theta >= 1.0:
        return q1
    
    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1 - cos_half_theta*cos_half_theta)
    
    if torch.abs(sin_half_theta) < 1e-6:
        return q1 * 0.5 + q2 * 0.5
    
    ratio_a = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratio_b = torch.sin(t * half_theta) / sin_half_theta
    
    return ratio_a * q1 + ratio_b * q2

def quat2rotm(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to rotation matrix"""
    w, x, y, z = q
    
    R = torch.tensor([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    
    return R

def pose2velc(timestamp: float, pose: torch.Tensor, dataset_name: str=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get velocity and angular velocity at specified timestamp
    
    Args:
        timestamp: Query timestamp
        pose: Pose data tensor [N, 8] (time,x,y,z,qx,qy,qz,qw)
        dataset_name: Name of dataset
        
    Returns:
        v: Linear velocity [3]
        omega: Angular velocity [3]
    """
    # Find nearest poses
    mask_pre = pose[:,0] < timestamp
    mask_post = pose[:,0] >= timestamp
    
    if not torch.any(mask_pre) or not torch.any(mask_post):
        raise ValueError("Timestamp out of bounds")
        
    time_pre = torch.where(mask_pre)[0][-1]
    time_after = torch.where(mask_post)[0][0]
    
    # Calculate velocity in world frame
    dt = pose[time_after,0] - pose[time_pre,0]
    v_mocap = (pose[time_after,1:4] - pose[time_pre,1:4])/dt
    
    # Interpolation parameter
    p = (timestamp - pose[time_pre,0])/dt
    
    # Get quaternions
    q1 = pose[time_pre,[7,4,5,6]]  # [w,x,y,z]
    q2 = pose[time_after,[7,4,5,6]]
    
    # Calculate angular velocity
    omega_mocap = quat2angvel(q1, q2, dt)
    
    # Interpolate rotation
    q = slerp(q1, q2, p)
    R = quat2rotm(q)
    
    # Transform velocities to body frame
    v_body = R @ v_mocap
    omega_body = R @ omega_mocap
    
    # Transform to specific camera frame
    if dataset_name == 'VECtor':
        T_lcam_body = torch.tensor([
            [-0.857137023976571, 0.03276713258773897, -0.5140451703406658, 0.09127742788053987],
            [0.01322063096422759, -0.9962462506036175, -0.08554895133864114, -0.02255409664008403],
            [-0.5149187674240416, -0.08012317505073682, 0.853486344222504, -0.02986309837992267],
            [0., 0., 0., 1.]
        ])
        
        T_lcam_lev = torch.tensor([
            [0.9999407352369797, 0.009183655542749752, 0.005846920950435052, 0.0005085820608404798],
            [-0.009131364645448854, 0.9999186289230431, -0.008908070070089353, -0.04081979450823404],
            [-0.005928253827254812, 0.008854151768176144, 0.9999432282899994, -0.0140781304960408],
            [0., 0., 0., 1.]
        ])
        
        T_lev_body = torch.linalg.solve(T_lcam_lev, T_lcam_body)
        R_lev_body = T_lev_body[:3,:3]
        
        v = R_lev_body @ v_body
        omega = R_lev_body @ omega_body
        
    elif dataset_name == 'MVSEC':
        v = v_mocap
        omega = omega_mocap
        
    else:
        # print("Warning: No Coordinates Transform Found")
        v = v_body
        omega = omega_body
        
    return v, omega

def vector_to_skew(vec):
    if len(vec.shape) == 1:
        vec = vec.unsqueeze(0)
        mat = kornia.geometry.conversions.vector_to_skew_symmetric_matrix(vec).squeeze(0)
    else:
        mat = kornia.geometry.conversions.vector_to_skew_symmetric_matrix(vec)
    return mat

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
            
        return kornia.geometry.camera.pixel2cam(
            depth, K_4x4_inv, pixels_batch)[..., :3]

class PoseOptimizer:
    def __init__(self, image_size, device="cuda"):
        self.H, self.W = image_size
        self.device = device
    
    def compute_B_matrix(self, x):
        # x shape: [N,3] normalized coordinate
        xx = x[:, 0]  # x coordinate
        yy = x[:, 1]  # y coordinate
        
        # B矩阵的两行
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
        v_skew = vector_to_skew(v_vec_tensor).squeeze(0)
        w_skew = vector_to_skew(w_vec_tensor).squeeze(0)
        
        s = 0.5 * (torch.mm(v_skew, w_skew) + torch.mm(w_skew, v_skew))
        
        # calculate v_skew @ x & s @ x
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
        
        # 计算B(x)ω
        B = self.compute_B_matrix(x_batch_tensor)  # [N,2,3]
        est_flow = (B @ w_vec_tensor.t()).squeeze(-1)

        # 计算误差
        error = u_batch_tensor[..., :2] - est_flow  # [N,2]
        return error.reshape(-1).unsqueeze(0)  # 展平误差向量
    
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
            w_dec = th.ScaleCostWeight(1.0)
            w_norm = th.ScaleCostWeight(0.0)  
            
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
            w_rot = th.ScaleCostWeight(1.0)
            
            rot_cost_fn = th.AutoDiffCostFunction(
                optim_vars,
                self.rotation_error_fn,
                x.shape[1] * 2,
                w_rot,
                aux_vars=aux_vars,
                name="rot_cost_fn"
            )
            
            return [rot_cost_fn]
    
    def optimize(self, normalized_coords, optical_flow, only_rotation=False, init_velc=None, num_iterations=1000):
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
            step_size=0.1,
        )
        
        if init_velc is not None:
            if only_rotation:
                w_init = init_velc[1] if len(init_velc) > 1 else init_velc[0]
            else:
                v_init, w_init = init_velc
        else:
            if not only_rotation:
                v_init = torch.randn((1,3), device=self.device)
                v_init = v_init / torch.norm(v_init, p=2, dim=-1, keepdim=True)
            w_init = torch.randn((1,3), device=self.device)
        
        theseus_inputs = {
            "norm_coords": normalized_coords,
            "optical_flow": optical_flow
        }
        
        if not only_rotation:
            theseus_inputs["linear_velocity"] = v_init
        theseus_inputs["angular_velocity"] = w_init
        
        # theseus_inputs = {
        #     "linear_velocity": v_init,
        #     "angular_velocity": w_init,
        #     "norm_coords": normalized_coords,
        #     "optical_flow": optical_flow
        # }
        
        theseus_optim = th.TheseusLayer(optimizer).to(self.device)
        with torch.no_grad():
            updated_inputs, info = theseus_optim.forward(
                theseus_inputs,
                optimizer_kwargs={
                    "track_best_solution": True,
                    "track_state_history": True,
                    "track_err_history": True,
                    "verbose": False,
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
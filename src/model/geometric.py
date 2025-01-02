import torch
import kornia
import theseus as th

from typing import Optional

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
        
    def create_cost_function(self, normalized_coords, optical_flow):
        
        v = th.Vector(dof=3, name="linear_velocity") 
        w = th.Vector(dof=3, name="angular_velocity")
        
        x = th.Variable(normalized_coords, name="norm_coords")  # [1,N,3]
        u = th.Variable(optical_flow, name="optical_flow")      # [1,N,3]
        
        def dec_error_fn(optim_vars, aux_vars):
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

        optim_vars = [v, w]
        aux_vars = [x, u]
        dec_cost_fn = th.AutoDiffCostFunction(
            optim_vars,
            dec_error_fn,
            x.shape[1],
            aux_vars=aux_vars,
            name="dec_cost_fn"
        )
        
        return dec_cost_fn
    
    def optimize(self, normalized_coords, optical_flow, num_iterations=1000):
        normalized_coords = normalized_coords.unsqueeze(0)
        optical_flow = optical_flow.unsqueeze(0)
        
        objective = th.Objective()
        cost_functions = self.create_cost_function(
            normalized_coords, optical_flow
        )
        objective.add(cost_functions)
            
        optimizer = th.GaussNewton(
            objective,
            max_iterations=num_iterations,
            step_size=1,
        )

        v_init = torch.ones((1,3), device=self.device)
        w_init = torch.ones((1,3), device=self.device)
        theseus_inputs = {
            "linear_velocity": v_init,
            "angular_velocity": w_init,
            "norm_coords": normalized_coords,
            "optical_flow": optical_flow
        }
        
        theseus_optim = th.TheseusLayer(optimizer).to(self.device)
        with torch.no_grad():
            updated_inputs, info = theseus_optim.forward(
                theseus_inputs,
                optimizer_kwargs={
                    "track_best_solution": True,
                    "verbose": True,
                },
            )
        
        v_opt = info.best_solution["linear_velocity"] 
        w_opt = info.best_solution["angular_velocity"]
        
        return v_opt, w_opt
import torch
import logging
import torch.nn as nn
from src.model.eventflow import EventFlowINR
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint
import random

EPS = 1e-5

logger = logging.getLogger(__name__)

class NeuralODEWarp:
    def __init__(self, flow_inr: EventFlowINR, device: torch.device, 
                 tref_setting: str, num_step: int, solver: str) -> None:
        self.flow_field = flow_inr       # calculate dy/dt
        self.device = device
        self.num_step = num_step
        self.tref_setting = tref_setting
        self.solver = solver
        
    def get_reference_time(self, batch_txy: torch.Tensor, tref_setting: str):
        """
        Get the value of t_ref based on the given setting.

        Args:
            batch_txy (torch.Tensor): A n*3 tensor [t,x,y].
            tref_setting (str): A string indicating the type of reference time to return.
                                Options: 'min', 'max', 'mid', 'random', 'multi'.

        Returns:
            torch.Tensor: A scalar or tensor based on tref_setting.
        """
        t_min = torch.min(batch_txy[:, 0]) + EPS
        t_max = torch.max(batch_txy[:, 0]) - EPS
        t_mid = (t_max + t_min) / 2
        
        if tref_setting == 'min':
            return t_min.unsqueeze(0)
        elif tref_setting == 'max':
            return t_max.unsqueeze(0)
        elif tref_setting == 'mid':
            return t_mid.unsqueeze(0)
        elif tref_setting == 'random':
            return t_min + (t_max - t_min) * random.uniform(0,1)  # Uniform sampling between t_min and t_max
        elif tref_setting == 'multi':
            return torch.stack([t_min, t_mid, t_max])  # Return all three values as a tensor
        else:
            logger.error("Invalid tref_setting. Choose from ['min', 'max', 'mid', 'random', 'multi']")
            raise ValueError("Invalid tref_setting. Choose from ['min', 'max', 'mid', 'random', 'multi']")
    
    def warp_events(self, batch_txy: torch.Tensor, t_ref: torch.Tensor, method: str = 'euler'):
        """
        Warp events to reference time using Neural ODE with choice of integration method.
        
        Args:
            batch_txy (torch.Tensor): A n*3 tensor [t,x,y].
            t_ref (torch.Tensor): t_ref can be a scalar[1] or a tensor of shape [1, n].
            method (str): Integration method - 'euler' or 'rk4'.
        
        Returns:
            torch.Tensor: Warped batch_txy.`    
        """
        batch_t0 = batch_txy[:, 0].unsqueeze(1)
        t_step = ((t_ref - batch_t0) / self.num_step).transpose(0, 1)
        num_warp = t_step.shape[0]
        warped_batch_txy = batch_txy.unsqueeze(0).repeat(num_warp, 1, 1)  # [1, n, 3] or [m, n, 3]

        def euler_step(current_state, dt):
            """Single Euler integration step."""
            pred_flow = self.flow_field.forward(current_state)  # [1, n, 2] or [m, n, 2]
            new_state = current_state.clone()
            new_state[..., 1:] += pred_flow * dt.unsqueeze(-1)
            # new_state[..., 1:] += pred_flow * torch.sign(dt.unsqueeze(-1))
            new_state[..., 0] += dt.squeeze()
            return new_state
        
        def rk4_step(current_state, dt):
            """Single RK4 integration step."""
            # k1 = f(t, y)
            k1 = self.flow_field.forward(current_state)
            
            # k2 = f(t + dt/2, y + dt*k1/2)
            k1_state = current_state.clone()
            k1_state[..., 1:] += (k1 * dt.unsqueeze(-1)) / 2
            k1_state[..., 0] += dt.squeeze() / 2
            k2 = self.flow_field.forward(k1_state)
            
            # k3 = f(t + dt/2, y + dt*k2/2)
            k2_state = current_state.clone()
            k2_state[..., 1:] += (k2 * dt.unsqueeze(-1)) / 2
            k2_state[..., 0] += dt.squeeze() / 2
            k3 = self.flow_field.forward(k2_state)
            
            # k4 = f(t + dt, y + dt*k3)
            k3_state = current_state.clone()
            k3_state[..., 1:] += k3 * dt.unsqueeze(-1)
            k3_state[..., 0] += dt.squeeze()
            k4 = self.flow_field.forward(k3_state)
            
            # y(t + dt) = y(t) + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            new_state = current_state.clone()
            flow_update = (k1 + 2*k2 + 2*k3 + k4) / 6
            new_state[..., 1:] += flow_update * dt.unsqueeze(-1)
            new_state[..., 0] += dt.squeeze()
            return new_state
        
        # Choose integration method
        step_function = rk4_step if method.lower() == 'rk4' else euler_step
        
        # Integration loop
        for _ in range(self.num_step):
            warped_batch_txy = step_function(warped_batch_txy, t_step)
        # print(f"After warp: {warped_batch_txy[0,10000]}")
        return warped_batch_txy

class ODEFunc(nn.Module):
    def __init__(self, flow_field: EventFlowINR):
        super().__init__()
        self.flow_field = flow_field
        
    def forward(self, t_norm: torch.Tensor, augmented_state: torch.Tensor):
        x = augmented_state[..., 0]
        y = augmented_state[..., 1]
        t0 = augmented_state[..., 2]
        delta_t = augmented_state[..., 3]
        
        t_real = t0 + t_norm * delta_t
        
        input_coords = torch.stack([
            t_real,
            x,
            y
        ], dim=-1)
        
        flow = self.flow_field(input_coords)
        
        dxdt_norm = flow[..., 0] * delta_t
        dydt_norm = flow[..., 1] * delta_t
        
        dt0dt = torch.zeros_like(t0)
        d_delta_t_dt = torch.zeros_like(delta_t)
        
        return torch.stack([
            dxdt_norm,
            dydt_norm,
            dt0dt,
            d_delta_t_dt
        ], dim=-1)

class NeuralODEWarpV2(nn.Module):
    def __init__(self, flow_inr: EventFlowINR, device: torch.device, 
                 tref_setting: str, solver: str = 'dopri5', rtol: float = 1e-7, 
                 atol: float = 1e-9) -> None:
        super().__init__()
        self.ode_func = ODEFunc(flow_inr)  
        self.flow_field = flow_inr
        self.device = device
        self.tref_setting = tref_setting
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        
    def get_reference_time(self, batch_txy: torch.Tensor, tref_setting: str):
        """
        Get the value of t_ref based on the given setting.

        Args:
            batch_txy (torch.Tensor): A n*3 tensor [t,x,y].
            tref_setting (str): A string indicating the type of reference time to return.
                                Options: 'min', 'max', 'mid', 'random', 'multi'.

        Returns:
            torch.Tensor: A scalar or tensor based on tref_setting.
        """
        t_min = torch.min(batch_txy[:, 0]) + EPS
        t_max = torch.max(batch_txy[:, 0]) - EPS
        t_mid = (t_max + t_min) / 2
        
        if tref_setting == 'min':
            return t_min.unsqueeze(0)
        elif tref_setting == 'max':
            return t_max.unsqueeze(0)
        elif tref_setting == 'mid':
            return t_mid.unsqueeze(0)
        elif tref_setting == 'random':
            return t_min + (t_max - t_min) * random.uniform(0,1)  # Uniform sampling between t_min and t_max
        elif tref_setting == 'multi':
            return torch.stack([t_min, t_mid, t_max])  # Return all three values as a tensor
        else:
            logger.error("Invalid tref_setting. Choose from ['min', 'max', 'mid', 'random', 'multi']")
            raise ValueError("Invalid tref_setting. Choose from ['min', 'max', 'mid', 'random', 'multi']")
    
    def _build_augmented_state(self, batch_txy: torch.Tensor, t_ref: torch.Tensor):
        t0 = batch_txy[:, 0]
        delta_t = t_ref - t0 
        
        spatial_coords = batch_txy[:, 1:]
        
        augmented_state = torch.cat([
            spatial_coords,
            t0.unsqueeze(-1),
            delta_t.unsqueeze(-1)
        ], dim=-1)
        
        return augmented_state
    
    def warp_events(self, batch_txy: torch.Tensor, t_ref: torch.Tensor):
        # t_ref = self.get_reference_time(batch_txy, tref_setting)
        
        augmented_init = self._build_augmented_state(batch_txy, t_ref)
        
        t_eval = torch.tensor([0.0, 1.0], device=self.device)
        
        solution = odeint(
            self.ode_func,
            augmented_init,
            t_eval,
            method=self.solver,
            options={
                "step_size": 1
            },
            rtol=self.rtol,
            atol=self.atol
        )
        
        final_state = solution[-1]
        warped_spatial = final_state[:, :2]
        
        return torch.cat([
            t_ref.expand(batch_txy.size(0)).unsqueeze(-1),  # 保持时间维度
            warped_spatial
        ], dim=-1)
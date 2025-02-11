import torch
import logging
from src.model.eventflow import EventFlowINR

EPS = 1e-5

logger = logging.getLogger(__name__)

class NeuralODEWarp:
    def __init__(self, flow_inr: EventFlowINR, device: torch.device, 
                 tref_setting: str, num_step: int, solver: str) -> None:
        self.flow_calculator = flow_inr       # calculate dy/dt
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
            return torch.clamp(t_min, min=0.0, max=1.0).unsqueeze(0)
        elif tref_setting == 'max':
            return torch.clamp(t_max, min=0.0, max=1.0).unsqueeze(0)
        elif tref_setting == 'mid':
            return torch.clamp(t_mid, min=0.0, max=1.0).unsqueeze(0)
        elif tref_setting == 'random':
            return t_min + (t_max - t_min) * torch.rand(1).to(self.device)  # Uniform sampling between t_min and t_max
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
            pred_flow = self.flow_calculator.forward(current_state)  # [1, n, 2] or [m, n, 2]
            new_state = current_state.clone()
            new_state[..., 1:] += pred_flow * dt.unsqueeze(-1)
            new_state[..., 0] += dt.squeeze()
            return new_state
        
        def rk4_step(current_state, dt):
            """Single RK4 integration step."""
            # k1 = f(t, y)
            k1 = self.flow_calculator.forward(current_state)
            
            # k2 = f(t + dt/2, y + dt*k1/2)
            k1_state = current_state.clone()
            k1_state[..., 1:] += (k1 * dt.unsqueeze(-1)) / 2
            k1_state[..., 0] += dt.squeeze() / 2
            k2 = self.flow_calculator.forward(k1_state)
            
            # k3 = f(t + dt/2, y + dt*k2/2)
            k2_state = current_state.clone()
            k2_state[..., 1:] += (k2 * dt.unsqueeze(-1)) / 2
            k2_state[..., 0] += dt.squeeze() / 2
            k3 = self.flow_calculator.forward(k2_state)
            
            # k4 = f(t + dt, y + dt*k3)
            k3_state = current_state.clone()
            k3_state[..., 1:] += k3 * dt.unsqueeze(-1)
            k3_state[..., 0] += dt.squeeze()
            k4 = self.flow_calculator.forward(k3_state)
            
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
        
        return warped_batch_txy
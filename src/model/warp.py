import torch
import logging
from .inr import EventFlowINR

EPS = 1e-3

logger = logging.getLogger(__name__)

class NeuralODEWarp:
    def __init__(self, flow_inr: EventFlowINR, device: torch.device, 
                 tref_setting: str, num_step: int) -> None:
        self.flow_calculator = flow_inr       # calculate dy/dt
        self.device = device
        self.num_step = num_step
        self.tref_setting = tref_setting

    # def get_reference_time(self, events: torch.Tensor, tref_setting: str):
    #     t_min, t_max = torch.min(events[:, 0]), torch.max(events[:, 0])
    #     t_mid = (t_min + t_max)/2
    #     if tref_setting == "max":
    #         return torch.clamp(t_max, min=0.0, max=1.0)
    #     elif tref_setting == "min":
    #         return torch.clamp(t_min, min=0.0, max=1.0)
    #     elif tref_setting == "mid":
    #         return torch.clamp(t_mid, min=0.0, max=1.0)
    #     elif tref_setting == "random":
    #         return t_min + (t_max - t_min) * torch.rand(1).to(self.device)
        
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
        t_min = torch.min(batch_txy[:, 0])
        t_max = torch.max(batch_txy[:, 0])
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
            
    # def warp_events(self, batch_txy: torch.Tensor, t_ref: torch.Tensor):
    #     batch_t0 = batch_txy[:, 0]  
    #     warped_batch_txy = batch_txy.clone()
    #     t_step: torch.Tensor = ((t_ref - batch_t0) / self.num_step).unsqueeze(1)
    #     # print(warped_events.dtype, t_step.dtype)
    #     for _ in range(self.num_step):
    #         pred_flow = self.flow_calculator.forward(warped_batch_txy)
    #         warped_batch_txy[:, 1:] += pred_flow * t_step
    #         warped_batch_txy[:, 0] += t_step.squeeze()
        return warped_batch_txy

    def warp_events(self, batch_txy: torch.Tensor, t_ref: torch.Tensor):
        """
        Warp events to reference time using Neural ODE (support multi-reference time)

        Args:
            batch_txy (torch.Tensor): A n*3 tensor [t,x,y].
            t_ref (torch.Tensor): t_ref can be a scalar[1] or a tensor of shape [1, n].

        Returns:
            torch.Tensor: Warped batch_txy.
        """
        batch_t0 = batch_txy[:, 0].unsqueeze(1)  
        t_step = ((t_ref - batch_t0) / self.num_step).transpose(0, 1)
        num_warp = t_step.shape[0]
        warped_batch_txy = batch_txy.unsqueeze(0).repeat(num_warp, 1, 1) # [1, n, 3] or [m, n, 3] 

        for _ in range(self.num_step):
            pred_flow = self.flow_calculator.forward(warped_batch_txy) # [1, n, 2] or [m, n, 2]
            warped_batch_txy[..., 1:] += pred_flow * t_step.unsqueeze(-1)
            warped_batch_txy[..., 0] += t_step.squeeze()

        return warped_batch_txy
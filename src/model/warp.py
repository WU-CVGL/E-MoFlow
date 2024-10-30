import torch

from .inr import EventFlowINR

EPS = 1e-3

class NeuralODEWarp:
    def __init__(self, flow_inr: EventFlowINR, device: torch.device) -> None:
        self.flow_calculator = flow_inr       # calculate dy/dt
        self.device = device
        self.num_step = 5

    def get_reference_time(self, events: torch.Tensor) -> dict:
        t_min, t_max = torch.min(events[:, 0]), torch.max(events[:, 0])
        return torch.clamp(t_max + EPS, min=0.0, max=1.0)
        # events_time = events[:, 0]
        # batch_xy0, batch_t0 = events[:, 1:], events[:, 0]
        # t_min, t_max = torch.min(batch_t0) - EPS, torch.max(batch_t0) + EPS
        # t_ref_1 = t_min * torch.rand(1).to(self.device)
        # t_ref_2 = t_max + (1 - t_max) * torch.rand(1).to(self.device)
        
        # choice = torch.randint(0, 2, (1,)).item()
        # if choice == 0:
        #     output = torch.clamp(t_min - EPS, min=0.0, max=1.0)
        # else: 
        #     output = torch.clamp(t_max + EPS, min=0.0, max=1.0)
        # return output

    def warp_events(self, batch_txy: torch.Tensor, t_ref):
        batch_t0 = batch_txy[:, 0]  
        warped_batch_txy = batch_txy.clone()
        t_step: torch.Tensor = ((t_ref - batch_t0) / self.num_step).unsqueeze(1)
        # print(warped_events.dtype, t_step.dtype)
        for _ in range(self.num_step):
            pred_flow = self.flow_calculator.forward(warped_batch_txy)
            warped_batch_txy[:, 1:] += pred_flow * t_step
            warped_batch_txy[:, 0] += t_step.squeeze()
        return warped_batch_txy





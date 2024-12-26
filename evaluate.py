import torch 
import cv2 as cv2
import numpy as np

from tqdm import tqdm
from pathlib import Path

from src.utils import misc
from src.utils import load_config
from src.model.inr import EventFlowINR
from src.utils.wandb import WandbLogger
from src.utils.visualizer import Visualizer
from src.model.geometric import KorniaPixel2Cam
from src.utils.event_image_converter import EventImageConverter

class DenseOpticalFlow:
    def __init__(self, grid_size, model, normalize_coords=True, device="cuda"):
        self.H, self.W = grid_size
        x, y = np.meshgrid(np.arange(self.W), np.arange(self.H))
        x = torch.tensor(x, dtype=torch.float32).view(-1)
        y = torch.tensor(y, dtype=torch.float32).view(-1)
        if normalize_coords:
            x = x / (self.W - 1)
            y = y / (self.H - 1)
        self.x = x.to(device)
        self.y = y.to(device)

        self.model = model
        self.device = device

    def __call__(self, t):
        num_points = self.x.shape[0]
        num_timesteps = t.shape[1]
        x_batch = self.x.repeat(num_timesteps)
        y_batch = self.y.repeat(num_timesteps)
        t_batch = t.view(-1).repeat_interleave(num_points).to(self.device)
        txy = torch.stack((t_batch, x_batch, y_batch), dim=1)
        
        with torch.no_grad():
            flow = self.model(txy)  # [num_points*num_timesteps, 2]
        
        u = flow[:, 0].view(num_timesteps, num_points)
        v = flow[:, 1].view(num_timesteps, num_points)
        U = u.view(num_timesteps, self.H, self.W)
        V = v.view(num_timesteps, self.H, self.W)
        
        U, V = U * (self.W - 1), V * (self.H - 1)
        
        return u, v, U, V

if __name__ == "__main__":
    args = load_config.parse_args()
    config = load_config.load_yaml_config(args.config)
    data_config = config["data"]
    model_config = config["model"]
    warp_config = config["warp"]
    optimizer_config = config["optimizer"]
    image_size = (data_config["hight"], data_config["weight"])

    # wandb
    wandb_logger = WandbLogger(config)

    # Visualizer
    viz = Visualizer(
        image_shape=image_size,
        show=False,
        save=True,
        save_dir=config["logger"]["results_dir"],
    )

    # detect cuda
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    # load model 
    flow_field = EventFlowINR(model_config).to(device)
    flow_field.load_state_dict(torch.load(config["logger"]["model_weight_path"]))
    flow_field.eval() 

    # dense optical flow at any t
    compute_dense_flow = DenseOpticalFlow(
        image_size, 
        flow_field, 
        normalize_coords=True, 
        device=device
    )

    t = torch.linspace(0, 1, steps=50).view(1, -1)  # shape: [1, d]
    u, v, U, V = compute_dense_flow(t)


    K_path = Path("/run/determined/workdir/ssd_data/Event_Dataset/Blender/final_motion_oneWall/K_matrix.txt")
    pixel2cam = KorniaPixel2Cam(image_size[0], image_size[1], device)
    K_tensor = misc.load_camera_intrinsic(K_path)
    normalized_pixel_grid = pixel2cam(K_tensor)

    for i in range(t.shape[1]):
        # visualize color optical flow
        color_flow, wheel = viz.visualize_optical_flow(
            flow_x=U[i].cpu().numpy(),
            flow_y=V[i].cpu().numpy(),
            visualize_color_wheel=True,
            file_prefix="test_dense_optical_flow",
            save_flow=False,
            ord=1.0,
        )

        arrow_flow = viz.visualize_flow_arrows(
            flow_x=U[i].cpu().numpy(),
            flow_y=V[i].cpu().numpy(),  
            file_prefix="optical_flow_arrow",
            sampling_ratio=0.001, 
            bg_color=(255, 255, 255)    
        )

        # upload wandb
        wandb_logger.write_img("color_optical_flow", color_flow)
        wandb_logger.write_img("color_wheel", wheel)
        wandb_logger.write_img("optical_flow_arrow", arrow_flow)
        wandb_logger.update_buffer()


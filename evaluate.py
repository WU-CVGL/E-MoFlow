import torch 
import cv2 as cv2
import numpy as np

from tqdm import tqdm

from src.utils import load_config
from src.model.inr import EventFlowINR
from src.utils.wandb import WandbLogger
from src.utils.visualizer import Visualizer
from src.utils.event_image_converter import EventImageConverter

def compute_dense_flow(
    model, t, 
    grid_size, 
    device="cuda",
    normalize_coords=True
):
    H, W = grid_size
    x, y = np.meshgrid(np.arange(W), np.arange(H))  
    x = torch.tensor(x, dtype=torch.float32).view(-1)  
    y = torch.tensor(y, dtype=torch.float32).view(-1)

    if normalize_coords:
        x = x / (W - 1)  
        y = y / (H - 1) 

    t = torch.full_like(x, t)  
    txy = torch.stack((t, x, y), dim=1).to(device) 

    with torch.no_grad():
        flow = model(txy)  
    u, v = flow[:, 0], flow[:, 1]
    U = u.view(H, W)
    V = v.view(H, W)

    return u, v, x, y, U, V

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
    t = 1 
    u, v, x, y, U, V = compute_dense_flow(flow_field, t, grid_size=image_size, device=device)

    # visualize color optical flow
    color_flow, wheel = viz.visualize_optical_flow(
        flow_x=U.cpu().numpy(),
        flow_y=V.cpu().numpy(),
        visualize_color_wheel=True,
        file_prefix="test_dense_optical_flow",
        save_flow=False,
        ord=1,
    )

    # upload wandb
    wandb_logger.write_img("color_optical_flow", color_flow)
    wandb_logger.write_img("color_wheel", wheel)
    wandb_logger.update_buffer()


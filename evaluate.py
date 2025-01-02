import torch 
import numpy as np

from tqdm import tqdm
from pathlib import Path

from src.utils import misc
from src.model import geometric
from src.utils import load_config
from src.model.inr import EventFlowINR
from src.utils.wandb import WandbLogger
from src.utils.visualizer import Visualizer
from src.model.flow import DenseOpticalFlowCalc

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
    flow_calculator = DenseOpticalFlowCalc(
        grid_size=image_size, 
        model=flow_field, 
        normalize_coords=True, 
        device=device
    )

    t = torch.linspace(0, 1, steps=50).view(1, -1)  # shape: [1, d]
    u, v, U, V = flow_calculator.extract_flow_from_inr(t)

    K_path = Path("/run/determined/workdir/ssd_data/Event_Dataset/Blender/final_motion_oneWall/K_matrix.txt")
    pixel2cam = geometric.Pixel2Cam(image_size[0], image_size[1], device)
    K_tensor = misc.load_camera_intrinsic(K_path)
    normalized_pixel_grid = pixel2cam(K_tensor.to(device))
    pose_optimizer = geometric.PoseOptimizer(image_size, device)

    for i in tqdm(range(t.shape[1])):
        current_flow = torch.stack([U[i], V[i]], dim=-1)
        current_flow = torch.nn.functional.pad(current_flow, (0, 1), mode='constant', value=0)
        current_coords = normalized_pixel_grid.squeeze(0).view(-1, 3)
        current_sparse_flow, indices = flow_calculator.sparsify_flow(
            current_flow,
            sparse_ratio=0.005,
            threshold=10
        )
        current_sparse_coords = current_coords[indices, :]
        v_est, w_est = pose_optimizer.optimize(current_sparse_coords, current_sparse_flow)
        print(f"linear_velocity:{v_est}, angular_velocity:{w_est}")
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


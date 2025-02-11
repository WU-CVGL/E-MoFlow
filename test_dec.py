import torch
import numpy as np

from tqdm import tqdm
from pathlib import Path

from src.utils import misc
from src.utils import pose
from src.model import geometric
from src.utils import vector_math
from src.utils import load_config
from src.model.eventflow import EventFlowINR
from src.utils.wandb import WandbLogger
from src.utils.visualizer import Visualizer
from src.model.eventflow import DenseOpticalFlowCalc

if __name__ == "__main__":
    args = load_config.parse_args()
    config = load_config.load_yaml_config(args.config)
    data_config = config["data"]
    model_config = config["model"]
    warp_config = config["warp"]
    optimizer_config = config["optimizer"]
    image_size = (data_config["hight"], data_config["weight"])

    misc.fix_random_seed()

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
    
    # load data
    K_path = Path("/run/determined/workdir/ssd_data/Event_Dataset/Blender/final_rotation_threeWall/K_matrix.txt")
    CamPose_path = Path("/run/determined/workdir/ssd_data/Event_Dataset/Blender/final_rotation_threeWall/camera_pose.txt")
    K_tensor = misc.load_camera_intrinsic(K_path)
    gt_camera_pose = misc.load_camera_pose(CamPose_path)

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
    
    # create optimizer
    pixel2cam = geometric.Pixel2Cam(image_size[0], image_size[1], device)
    normalized_pixel_grid = pixel2cam(K_tensor.to(device))
    pose_optimizer = geometric.PoseOptimizer(image_size, device)

    # valid using gt depth
    DEPTH_GT = np.loadtxt('/run/determined/workdir/ssd_data/Event_Dataset/Blender/final_rotation_threeWall/depth/00000001.txt')
    DEPTH_GT = torch.from_numpy(DEPTH_GT).reshape(480,640,1).to(device)
    
    # valid using gt optical flow
    U_GT = np.loadtxt('/run/determined/workdir/ssd_data/Event_Dataset/Blender/final_rotation_threeWall/forward_flow/00000001_x.txt')
    V_GT = np.loadtxt('/run/determined/workdir/ssd_data/Event_Dataset/Blender/final_rotation_threeWall/forward_flow/00000001_y.txt')
    U_GT = torch.from_numpy(U_GT).reshape(480,640)
    V_GT = torch.from_numpy(V_GT).reshape(480,640)
    
    current_flow = torch.stack([U_GT, V_GT], dim=-1).float().to(device)
    current_flow = torch.nn.functional.pad(current_flow, (0, 1), mode='constant', value=0)
    
    v_gt, w_gt = pose.pose_to_velocity(0.001, gt_camera_pose)
    v_gt, w_gt = v_gt.unsqueeze(0).to(device), w_gt.unsqueeze(0).to(device)
    
    normalized_pixel_grid = normalized_pixel_grid.squeeze(0)
    optical_flow = geometric.compute_motion_field(normalized_pixel_grid, v_gt, w_gt, DEPTH_GT)
    
    current_coords = normalized_pixel_grid.view(-1,3)
    
    current_sparse_flow, indices = flow_calculator.sparsify_flow(
        current_flow,
        sparse_ratio=0.01,
        threshold=0.0001
    )
    current_sparse_coords = current_coords[indices, :]

    arrow_gt_flow = viz.visualize_flow_arrows(
            flow_x=(U_GT*908.72409/10).cpu().numpy(),
            flow_y=(V_GT*908.72409/10).cpu().numpy(),  
            file_prefix="optical_flow_arrow",
            sampling_ratio=0.001, 
            bg_color=(255, 255, 255)    
    )
    
    arrow_flow = viz.visualize_flow_arrows(
            flow_x=(optical_flow[...,0]*908.72409/10).cpu().numpy(),
            flow_y=(optical_flow[...,1]*908.72409/10).cpu().numpy(),  
            file_prefix="optical_flow_arrow",
            sampling_ratio=0.001, 
            bg_color=(255, 255, 255)    
    )
    
    wandb_logger.write_img("gt_flow_arrow", arrow_gt_flow)
    wandb_logger.write_img("flow_arrow", arrow_flow)
    wandb_logger.update_buffer()
    
    noise_level = 100.0
    v_gt_noisy = v_gt + noise_level * torch.randn_like(v_gt, device=v_gt.device)
    w_gt_noisy = w_gt + noise_level * torch.randn_like(w_gt, device=w_gt.device)
    v_gt_noisy = v_gt_noisy / torch.norm(v_gt_noisy, p=2, dim=-1, keepdim=True)
    init_velc = [v_gt_noisy, w_gt_noisy]
    
    # v_est, w_est, v_his, w_his, err_his = pose_optimizer.optimize(current_sparse_coords, current_sparse_flow, only_rotation=False, init_velc=init_velc)
    _, w_est, _, w_his, err_his = pose_optimizer.optimize(current_sparse_coords, current_sparse_flow, only_rotation=True, init_velc=init_velc)   
    # v_est = v_est / torch.norm(v_est, p=2, dim=-1, keepdim=True)
    # print(f"======================================================= Pose Optimization =======================================================")
    # print(f"groundtruth_linear_velocity:{v_gt}, groundtruth_angular_velocity:{w_gt}")
    # print(f"initial_linear_velocity:{init_velc[0]}, intial_angular_velocity:{init_velc[1]}")
    # print(f"estimated_linear_velocity:{v_est}, estimated_angular_velocity:{w_est}")
    # print(f"linear_velocity_error:{vector_math.compute_vector_angle_in_degrees(v_est.to(device), v_gt)}, angular_velocity_error:{vector_math.compute_vector_angle_in_degrees(w_est.to(device), w_gt)}")
    # print(f"dec_last_error:{err_his[-1]}")
    
    print(f"======================================================= Pose Optimization =======================================================")
    print(f"groundtruth_angular_velocity:{w_gt}")
    print(f"intial_angular_velocity:{init_velc[1]}")
    print(f"estimated_angular_velocity:{w_est}")
    print(f"angular_velocity_error:{vector_math.compute_vector_angle_in_degrees(w_est.to(device), w_gt)}")
    print(f"dec_last_error:{err_his[-1]}")
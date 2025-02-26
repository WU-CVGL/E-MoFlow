import torch
import numpy as np
import imageio.v2 as imageio

from tqdm import tqdm
from pathlib import Path

from src.utils import (
    misc, 
    pose, 
    vector_math,
    load_config,
    metric
)

from src.model import geometric
from src.model.eventflow import EventFlowINR
from src.utils.wandb import WandbLogger
from src.utils.visualizer import Visualizer
from src.model.eventflow import DenseOpticalFlowCalc
from src.loader import dataset_manager
from src.utils.event_image_converter import EventImageConverter

torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    args = load_config.parse_args()
    config = load_config.load_yaml_config(args.config)
    data_config = config["data"]
    model_config = config["model"]
    warp_config = config["warp"]
    optimizer_config = config["optimizer"]
    image_size = (data_config["hight"], data_config["weight"])

    misc.fix_random_seed()
    
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
    
    # load data
    K_path = Path("/run/determined/workdir/ssd_data/Event_Dataset/Blender/final_motion_oneWall/K_matrix.txt")
    CamPose_path = Path("/run/determined/workdir/ssd_data/Event_Dataset/Blender/final_motion_oneWall/camera_pose.txt")
    TimeStamps_path = Path("/run/determined/workdir/ssd_data/Event_Dataset/Blender/final_motion_oneWall/timestamp.txt")
    depth_folder = '/run/determined/workdir/ssd_data/Event_Dataset/Blender/final_motion_oneWall/depth'
    events_folder = '/run/determined/workdir/ssd_data/Event_Dataset/Blender/final_motion_oneWall/events'
    K_tensor = misc.load_camera_intrinsic(K_path)
    gt_camera_pose = misc.load_camera_pose(CamPose_path)
    timestamps = misc.load_time_stamps(TimeStamps_path)
    depth_paths = misc.get_sorted_txt_paths(depth_folder)
    events_paths = misc.get_sorted_txt_paths(events_folder)
    
    # event2img converter
    image_size = (data_config["hight"], data_config["weight"])
    converter = EventImageConverter(image_size)
    
    # load model 
    flow_field = EventFlowINR(model_config).to(device)
    flow_field.load_state_dict(torch.load(config["logger"]["model_weight_path"]))
    flow_field.eval() 

    # dense optical flow at any t
    flow_calculator = DenseOpticalFlowCalc(
        grid_size=image_size, 
        intrinsic_mat = K_tensor,
        model=flow_field, 
        normalize_coords_mode="NORM_PLANE", 
        device=device
    )
    sequence_length = 100
    t = torch.linspace(0, 1, steps=sequence_length).view(1, -1).to(device)  # shape: [1, d]
    # print(len(t[0]))
    # print(timestamps[sequence_length-1])
    time_scale = 1 / timestamps[sequence_length - 1]
    # print(t[0,-1] / time_scale) 
    # print(t[0,-2] / time_scale) 
    # time_scale = 1
    # t_mid = ((t[0, 1:] + t[0, :-1]) / 2).unsqueeze(0) 
    # U, V, U_norm, V_norm = flow_calculator.extract_flow_from_inr(t, time_scale)

    pixel2cam = geometric.Pixel2Cam(image_size[0], image_size[1], device)
    normalized_pixel_grid = pixel2cam.generate_normalized_coordinate(K_tensor.to(device))
    normalized_pixel_grid = normalized_pixel_grid.squeeze(0)
    pose_optimizer = geometric.PoseOptimizer(image_size, device)

    for i in tqdm(range(sequence_length)):
        events = torch.from_numpy(np.loadtxt(events_paths[i])).to(device)
        events = events[:, [2,1,0,3]]
        events_mask = converter.create_eventmask(events)
        events_mask = events_mask[0].squeeze(0)
        
        depth_gt = np.loadtxt(depth_paths[i])
        depth_gt = torch.from_numpy(depth_gt).reshape(480,640,1).to(device)
        if(i == 0):
            v_gt, w_gt = pose.pose_to_velocity(0.001, gt_camera_pose)
        else:
            v_gt, w_gt = pose.pose_to_velocity(timestamps[i], gt_camera_pose)
        v_gt, w_gt = v_gt.unsqueeze(0).to(device), w_gt.unsqueeze(0).to(device)
        
        gt_optical_flow_norm, gt_optical_flow = geometric.compute_motion_field(K_tensor, normalized_pixel_grid, v_gt, w_gt, depth_gt)

        U, V, U_norm, V_norm = flow_calculator.extract_flow_from_inr(t[...,i], time_scale)
        stacked = torch.stack((U_norm, V_norm), dim=2)
        zeros = torch.zeros(stacked.shape[:2] + (1,), dtype=stacked.dtype).to(device)
        pred_optical_flow_norm = torch.cat((stacked, zeros), dim=2)
        # print(pred_optical_flow_norm.shape)
        current_coords = normalized_pixel_grid.view(-1,3)
        current_sparse_flow, indices = flow_calculator.sparsify_flow(
            pred_optical_flow_norm,
            sparse_ratio=0.01,
            threshold=0.0001
        )
        current_sparse_coords = current_coords[indices, :]

        noise_level = 0.0
        v_gt_noisy = v_gt + noise_level * torch.randn_like(v_gt, device=v_gt.device)
        w_gt_noisy = w_gt + noise_level * torch.randn_like(w_gt, device=w_gt.device)
        v_gt_noisy = v_gt_noisy / torch.norm(v_gt_noisy, p=2, dim=-1, keepdim=True)
        init_velc = [v_gt_noisy, w_gt_noisy]
        
        v_est, w_est, v_his, w_his, err_his = pose_optimizer.optimize(
            current_sparse_coords, 
            current_sparse_flow, 
            only_rotation=False, 
            init_velc=init_velc,
            num_iterations=1000
        )
        # _, w_est, _, w_his, err_his = pose_optimizer.optimize(current_sparse_coords, current_sparse_flow, only_rotation=True, init_velc=init_velc)   
        # v_est = v_est / torch.norm(v_est, p=2, dim=-1, keepdim=True)
        v_gt_norm = v_gt / torch.norm(v_gt, p=2, dim=-1, keepdim=True)
        # print(v_his)
        v_dir_error = vector_math.vector_dir_error_in_degrees(v_est.to(device), v_gt_norm)
        w_dir_error = vector_math.vector_dir_error_in_degrees(w_est.to(device), w_gt)
        v_mag_error = vector_math.vector_mag_error(v_est.to(device), v_gt_norm)
        w_mag_error = vector_math.vector_mag_error(w_est.to(device), w_gt)
        
        print(f"======================================================= Pose Optimization =======================================================")
        print(f"groundtruth_linear_velocity:{v_gt}, groundtruth_angular_velocity:{w_gt}")
        print(f"initial_linear_velocity:{init_velc[0]}, intial_angular_velocity:{init_velc[1]}")
        print(f"estimated_linear_velocity:{v_est}, estimated_angular_velocity:{w_est}")
        print(f"linear_velocity_dir_error:{v_dir_error}, angular_velocity_dir_error:{w_dir_error}")
        print(f"linear_velocity_mag_error:{v_mag_error}, angular_velocity_mag_error:{w_mag_error}")
        
        # visualize color optical flow
        eval_color_flow, wheel = viz.visualize_optical_flow(
            flow_x=U.cpu().numpy(),
            flow_y=V.cpu().numpy(),
            visualize_color_wheel=True,
            file_prefix="evaluate_dense_optical_flow",
            save_flow=False,
            ord=0.5,
        )

        eval_arrow_flow = viz.visualize_flow_arrows(
            flow_x=(U / 10).cpu().numpy(),
            flow_y=(V / 10).cpu().numpy(),  
            file_prefix="evaluate_optical_flow_arrow",
            sampling_ratio=0.001, 
            bg_color=(255, 255, 255)    
        )
        
        gt_color_flow, wheel = viz.visualize_optical_flow(
            flow_x=(gt_optical_flow[...,0]).cpu().numpy(),
            flow_y=(gt_optical_flow[...,1]).cpu().numpy(),
            visualize_color_wheel=True,
            file_prefix="gt_dense_optical_flow",
            save_flow=False,
            ord=0.5,
        )
        
        gt_arrow_flow = viz.visualize_flow_arrows(
            flow_x=(gt_optical_flow[...,0]/10).cpu().numpy(),
            flow_y=(gt_optical_flow[...,1]/10).cpu().numpy(),  
            file_prefix="gt_optical_flow_arrow",
            sampling_ratio=0.001, 
            bg_color=(255, 255, 255)    
        )

        # metric
        gt_optical_flow = gt_optical_flow[..., :2].permute(2, 0, 1).unsqueeze(0)
        pred_optical_flow = torch.stack([U, V], dim=2).permute(2, 0, 1).unsqueeze(0)
        error, endpoint_error = metric.calculate_flow_error(
            flow_gt=gt_optical_flow, 
            flow_pred=pred_optical_flow,
            event_mask=events_mask,
            time_scale=None
        )
        
        error_map = metric.visualize_endpoint_error(endpoint_error)
        # misc.save_flow("./outputs/flowmap/boxes_gt.png", gt_optical_flow.cpu().numpy())
        
        # upload wandb
        wandb_logger.write("EPE", error["EPE"].item())
        wandb_logger.write("AE", error["AE"].item())
        wandb_logger.write("1PE", error["1PE"].item())
        wandb_logger.write("2PE", error["2PE"].item())
        wandb_logger.write("3PE", error["3PE"].item())
        wandb_logger.write("v_dir_error", v_dir_error)
        wandb_logger.write("w_dir_error", w_dir_error)
        wandb_logger.write("v_mag_error", v_mag_error)
        wandb_logger.write("w_mag_error", w_mag_error)
        wandb_logger.write_img("events_mask", events_mask.cpu().numpy() * 255)
        wandb_logger.write_img("error_map", error_map.cpu().numpy())
        wandb_logger.write_img("evaluate_dense_optical_flow", eval_color_flow)
        wandb_logger.write_img("evaluate_optical_flow_arrow", eval_arrow_flow)
        wandb_logger.write_img("gt_dense_optical_flow", gt_color_flow)
        wandb_logger.write_img("gt_optical_flow_arrow", gt_arrow_flow)
        wandb_logger.write_img("color_wheel", wheel)
        wandb_logger.update_buffer()
        
        
    


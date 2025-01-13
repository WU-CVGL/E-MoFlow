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
    
    # load data
    K_path = Path("/run/determined/workdir/ssd_data/Event_Dataset/Blender/final_motion_boxes/K_matrix.txt")
    CamPose_path = Path("/run/determined/workdir/ssd_data/Event_Dataset/Blender/final_motion_boxes/camera_pose.txt")
    K_tensor = misc.load_camera_intrinsic(K_path)
    gt_camera_pose = misc.load_camera_pose(CamPose_path)
    
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
    t_mid = ((t[0, 1:] + t[0, :-1]) / 2).unsqueeze(0) 
    u, v, U, V = flow_calculator.extract_flow_from_inr(t_mid)

    # create optimizer
    pixel2cam = geometric.Pixel2Cam(image_size[0], image_size[1], device)
    normalized_pixel_grid = pixel2cam(K_tensor.to(device))
    pose_optimizer = geometric.PoseOptimizer(image_size, device)

    # valid using gt optical flow
    U_GT = np.loadtxt('/run/determined/workdir/ssd_data/Event_Dataset/Blender/final_motion_boxes/forward_flow/00000000_x.txt')
    V_GT = np.loadtxt('/run/determined/workdir/ssd_data/Event_Dataset/Blender/final_motion_boxes/forward_flow/00000000_y.txt')
    U_GT = torch.from_numpy(U_GT).reshape(480,640)
    V_GT = torch.from_numpy(V_GT).reshape(480,640)
    current_flow = torch.stack([U_GT, V_GT], dim=-1).float().to(device)
    current_flow = torch.nn.functional.pad(current_flow, (0, 1), mode='constant', value=0)
    current_coords = normalized_pixel_grid.squeeze(0).view(-1, 3)

    current_sparse_flow, indices = flow_calculator.sparsify_flow(
        current_flow,
        sparse_ratio=0.01,
        threshold=0.0001
    )
    current_sparse_coords = current_coords[indices, :]
    
    v_gt, w_gt = geometric.pose2velc(0.001, gt_camera_pose)
    v_gt, w_gt = v_gt.unsqueeze(0).to(device), w_gt.unsqueeze(0).to(device)

    noise_level = 0.0
    v_gt_noisy = v_gt + noise_level * torch.randn_like(v_gt, device=v_gt.device)
    w_gt_noisy = w_gt + noise_level * torch.randn_like(w_gt, device=w_gt.device)
    init_velc = [v_gt_noisy, w_gt_noisy]
    # init_velc = [v_gt, w_gt]
    # init_velc = [v_gt.unsqueeze(0).to(device), w_gt.unsqueeze(0).to(device)]
    # init_velc = None
    v_est, w_est, v_his, w_his, err_his = pose_optimizer.optimize(current_sparse_coords, current_sparse_flow, only_rotation=False, init_velc=init_velc)
    # _, w_est, _, w_his, err_his = pose_optimizer.optimize(current_sparse_coords, current_sparse_flow, only_rotation=True, init_velc=init_velc)   
    # v_est = v_est / torch.norm(v_est, p=2, dim=-1, keepdim=True)
    print(f"======================================================= Pose Optimization =======================================================")
    print(f"groundtruth_linear_velocity:{v_gt}, groundtruth_angular_velocity:{w_gt}")
    print(f"initial_linear_velocity:{init_velc[0]}, intial_angular_velocity:{init_velc[1]}")
    print(f"estimated_linear_velocity:{v_est}, estimated_angular_velocity:{w_est}")
    print(f"linear_velocity_error:{geometric.vector_angle_degree(v_est.to(device), v_gt)}, angular_velocity_error:{geometric.vector_angle_degree(w_est.to(device), w_gt)}")
    print(f"dec_last_error:{err_his[-1]}")
    # for i in tqdm(range(t_mid.shape[1])):
    #     # prepare flow and coordinate data
    #     current_flow = torch.stack([U[i], V[i]], dim=-1)
    #     current_flow = torch.nn.functional.pad(current_flow, (0, 1), mode='constant', value=0)
    #     current_coords = normalized_pixel_grid.squeeze(0).view(-1, 3)
    #     current_sparse_flow, indices = flow_calculator.sparsify_flow(
    #         current_flow,
    #         sparse_ratio=0.01,
    #         threshold=0.0001
    #     )
    #     current_sparse_coords = current_coords[indices, :]
        
    #     # get gt velocity
    #     mid_timestamp = ((gt_camera_pose[i][0] + gt_camera_pose[i+1][0]) / 2.0).item()
    #     v_gt, w_gt = geometric.pose2velc(mid_timestamp, gt_camera_pose)
    #     v_gt, w_gt = v_gt.unsqueeze(0).to(device), w_gt.unsqueeze(0).to(device)
        
    #     noise_level = 100.0
    #     v_gt_noisy = v_gt + noise_level * torch.randn_like(v_gt, device=v_gt.device)
    #     w_gt_noisy = w_gt + noise_level * torch.randn_like(w_gt, device=w_gt.device)
    #     init_velc = [v_gt_noisy, w_gt_noisy]
    #     # optimize velocity 
    #     # init_velc = [v_gt, w_gt]
    #     # init_velc = None
    #     # v_est, w_est, v_his, w_his, err_his = pose_optimizer.optimize(current_sparse_coords, current_sparse_flow, only_rotation=False, init_velc=init_velc)
    #     _, w_est, _, w_his, err_his = pose_optimizer.optimize(current_sparse_coords, current_sparse_flow, only_rotation=True, init_velc=init_velc)   
    #     # v_est = v_est / torch.norm(v_est, p=2, dim=-1, keepdim=True)
    #     # print(f"=============================================== ITER {i} ====================================================")
    #     # print(f"groundtruth_linear_velocity:{v_gt}, groundtruth_angular_velocity:{w_gt}")
    #     # print(f"estimated_linear_velocity:{v_est}, estimated_angular_velocity:{w_est}")
    #     # print(f"linear_velocity_error:{geometric.vector_angle_degree(v_est, v_gt)}, linear_velocity_error:{geometric.vector_angle_degree(w_est, w_gt)}")
        
    #     print(f"==================================================== Pose Optimization {i} ====================================================")
    #     print(f"groundtruth_linear_velocity:{v_gt}, groundtruth_angular_velocity:{w_gt}")
    #     print(f"intial_angular_velocity:{init_velc[1]}")
    #     print(f"estimated_linear_velocity:None, estimated_angular_velocity:{w_est}")
    #     print(f"linear_velocity_error:None, angular_velocity_error:{geometric.vector_angle_degree(w_est.to(device), w_gt)}")
    
    #     # visualize color optical flow
    #     color_flow, wheel = viz.visualize_optical_flow(
    #         flow_x=U[i].cpu().numpy(),
    #         flow_y=V[i].cpu().numpy(),
    #         visualize_color_wheel=True,
    #         file_prefix="test_dense_optical_flow",
    #         save_flow=False,
    #         ord=1.0,
    #     )

    #     arrow_flow = viz.visualize_flow_arrows(
    #         flow_x=U[i].cpu().numpy(),
    #         flow_y=V[i].cpu().numpy(),  
    #         file_prefix="optical_flow_arrow",
    #         sampling_ratio=0.001, 
    #         bg_color=(255, 255, 255)    
    #     )

    #     # upload wandb
    #     wandb_logger.write_img("color_optical_flow", color_flow)
    #     wandb_logger.write_img("color_wheel", wheel)
    #     wandb_logger.write_img("optical_flow_arrow", arrow_flow)
    #     wandb_logger.update_buffer()
        


import os
import sys
import torch 
import logging
import cv2 as cv2
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from src.loss.focus import FocusLoss
from src.loss.motion import VelocityLoss
from src.utils import misc
from src.utils import pose
from src.utils import event_proc
from src.utils import load_config
from src.model.eventflow import EventFlowINR
from src.utils.wandb import WandbLogger
from src.model.warp import NeuralODEWarp
from src.model import geometric
from src.utils.timer import TimeAnalyzer
from src.dataset_loader import dataset_manager
from src.model.eventflow import DenseOpticalFlowCalc
from src.utils.event_image_converter import EventImageConverter

# log
logging.basicConfig(
    handlers=[
        logging.FileHandler(f"main.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # load configs
    args = load_config.parse_args()
    config = load_config.load_yaml_config(args.config)
    data_config = config["data"]
    model_config = config["model"]
    warp_config = config["warp"]
    optimizer_config = config["optimizer"]

    # wandb
    wandb_logger = WandbLogger(config)

    # detect cuda
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    logger.info(f"Use device: {device}")

    # fix seed
    misc.fix_random_seed()

    # load dataset 
    dataset_name = data_config.pop('dataset_name')
    dataloader_config = data_config.pop('loader')
    dataset = dataset_manager.get_dataset(dataset_name, data_config)
    loader, provider = dataset_manager.create_loader(dataset, dataloader_config)
    camera_K = provider.get_camera_K().to(device)
    gt_camera_pose = provider.get_gt_camera_pose().to(device)
    [start_index, end_index] = provider.sequence_indices
    end_time, start_time = gt_camera_pose[...,0][end_index], gt_camera_pose[...,0][start_index] 
    time_scale = torch.Tensor(1 / (end_time - start_time)).to(device)
    
    # event2img converter
    image_size = (data_config["hight"], data_config["weight"])
    converter = EventImageConverter(image_size)
    
    # create the grid of camera normalized plane
    pixel2cam = geometric.Pixel2Cam(image_size[0], image_size[1], device)
    norm_coords = pixel2cam.generate_normalized_coordinate(camera_K.to(device))
    
    # create model
    flow_field: EventFlowINR = EventFlowINR(model_config).to(device)
    warpper = NeuralODEWarp(flow_field, device, **warp_config)
    
    # create NN optimizer
    optimizer = optim.Adam(flow_field.parameters(), lr=optimizer_config["initial_lrate"])
    
    # create theseus optimizer
    pose_optimizer = geometric.PoseOptimizer(image_size, device)

    # instantiate criterion
    grad_criterion = FocusLoss(loss_type="gradient_magnitude", norm="l1")
    var_criterion = FocusLoss(loss_type="variance", norm="l1")
    motion_criterion = VelocityLoss(alpha=0.4, beta=0.4, gamma=0.8)
    
    # prepare data to valid
    valid_data = provider.get_valid_data(config["valid"]["file_idx"])
    valid_events = valid_data["events"]
    valid_events_norm = valid_data["events_norm"]
    valid_events_timestamps = valid_data["timestamps"]
    valid_norm_txy = valid_events_norm[:, :-1].float().to(device)

    # display origin valid data
    valid_origin_iwe = converter.create_iwes(valid_events[:, [2,1,0,3]])
    wandb_logger.write_img("iwe", valid_origin_iwe.detach().cpu().numpy() * 255)
    wandb_logger.update_buffer()

    # time analysis
    time_analyzer = TimeAnalyzer()

    # train
    num_epochs = optimizer_config["num_epoch"]
    decay_steps = 1
    decay_factor = (optimizer_config["final_lrate"] / optimizer_config["initial_lrate"]) ** (1 / (num_epochs / decay_steps))
    for i in range(num_epochs):
        
        time_analyzer.start_epoch()

        for idx, sample in enumerate(tqdm(loader, desc=f"Tranning {i} epoch", leave=True)):
            optimizer.zero_grad()
            
            # get batch data
            events = sample["events"].squeeze(0)
            events_norm = sample["events_norm"].squeeze(0)
            timestamps = sample["timestamps"].squeeze(0)
            norm_txy = events_norm[:, :-1].float().to(device)
            origin_iwe = converter.create_iwes(events[:, [2,1,0,3]])

            # get t_ref
            t_ref = warpper.get_reference_time(norm_txy, warp_config["tref_setting"])

            # warp by neural ode 
            warped_norm_txy = warpper.warp_events(norm_txy, t_ref, method=warp_config["solver"])

            # create image warped event    
            num_iwe = warped_norm_txy.shape[0]
            num_events = warped_norm_txy.shape[1]
            polarity = events[:, 3].unsqueeze(0).expand(num_iwe, num_events).unsqueeze(-1).to(device)
            warped_norm_xytp = torch.cat((warped_norm_txy[..., [1,2,0]], polarity), dim=2)
            warped_events = event_proc.normalized_plane_to_pixel(
                warped_norm_xytp, 
                camera_K
            )
            # (x,y,t,p) ——> (y,x,t,p)
            warped_events = warped_events[...,[1,0,2,3]]
            # events should be reorganized as (y,x,t,p) that can processed by create_iwes
            iwes = converter.create_iwes(
                events=warped_events,
                method="bilinear_vote",
                sigma=1.0,
                blur=True
            ) # [n,h,w] n can be one or multi

            # CMax loss
            if num_iwe == 1:
                # var_loss = torch.Tensor([0]).to(device)
                var_loss = var_criterion(iwes)
                grad_loss = grad_criterion(iwes)
                # var_loss = focus.calculate_focus_loss(iwes, loss_type='variance')
                # grad_loss = focus.calculate_focus_loss(iwes, loss_type='gradient_magnitude', norm='l1')
            elif num_iwe > 1:
                var_loss = torch.Tensor([0]).to(device)
                # var_loss = focus.calculate_focus_loss(iwe, loss_type='variance')
                # grad_loss_t_min = focus.calculate_focus_loss(iwes[0].unsqueeze(0), loss_type='gradient_magnitude', norm='l1')
                # grad_loss_t_mid = focus.calculate_focus_loss(iwes[len(iwes)//2].unsqueeze(0), loss_type='gradient_magnitude', norm='l1')
                # grad_loss_t_max = focus.calculate_focus_loss(iwes[-1].unsqueeze(0), loss_type='gradient_magnitude', norm='l1')
                # grad_loss_origin = focus.calculate_focus_loss(origin_iwe, loss_type='gradient_magnitude', norm='l1')

                grad_loss_t_min = grad_criterion(iwes[0].unsqueeze(0))
                grad_loss_t_mid = grad_criterion(iwes[len(iwes)//2].unsqueeze(0))
                grad_loss_t_max = grad_criterion(iwes[-1].unsqueeze(0))
                grad_loss_origin = grad_criterion(origin_iwe)
                
                grad_loss = (grad_loss_t_min + grad_loss_t_max + 2 * grad_loss_t_mid) / 4 * grad_loss_origin
            
            # sample coordinates
            sample_coords = pixel2cam.sample_sparse_points(sparsity_level=(48*64), norm_coords=norm_coords)
            t_ref_expanded = t_ref * torch.ones(sample_coords.shape[1], device=device).reshape(1,-1,1).to(device)
            sample_norm_txy = torch.cat((t_ref_expanded, sample_coords[...,0:2]), dim=2)
            
            # get optical flow
            sample_flow = flow_field.forward(sample_norm_txy)
            sample_flow = F.pad(sample_flow, (0, 1), mode='constant', value=0)
            sample_flow = sample_flow * time_scale # scale
            
            # get gt velocity
            v_gt, w_gt = pose.pose_to_velocity(t_ref / time_scale, gt_camera_pose)
            v_gt, w_gt = v_gt.unsqueeze(0).to(device), w_gt.unsqueeze(0).to(device)
            v_gt = v_gt / torch.norm(v_gt, p=2, dim=-1, keepdim=True)
            
            # intial value for differential epipolar constrain
            noise_level = 0.0
            v_gt_noisy = v_gt + noise_level * torch.randn_like(v_gt, device=v_gt.device)
            w_gt_noisy = w_gt + noise_level * torch.randn_like(w_gt, device=w_gt.device)
            # v_gt_noisy = v_gt_noisy / torch.norm(v_gt_noisy, p=2, dim=-1, keepdim=True)
            init_velc = [v_gt_noisy, w_gt_noisy]
            
            # theseus optimize 
            v_opt, w_opt, v_his, w_his, err_his = pose_optimizer.optimize(
                sample_coords, sample_flow, 
                only_rotation=False, 
                init_velc=init_velc,
                num_iterations=100
            )

            # Motion loss
            motion_loss = motion_criterion(w_opt.to(device), v_opt.to(device), w_gt, v_gt)
            # motion_loss = torch.Tensor([0]).to(device)
            
            # Total loss
            alpha, beta, gamma = 1, 1, 1
            scaled_grad_loss =  alpha * grad_loss / 8.0
            scaled_var_loss = beta * var_loss / 31.0
            scaled_motion_loss = gamma * motion_loss
            total_loss = - scaled_grad_loss - scaled_var_loss + scaled_motion_loss
            
            wandb_logger.write("var_loss", scaled_var_loss.item())
            wandb_logger.write("grad_loss", scaled_grad_loss.item())
            wandb_logger.write("motion_loss", scaled_motion_loss.item())
            wandb_logger.write("train_loss", total_loss.item())

            # NN optimize
            total_loss.backward()
            optimizer.step()
            
        # print loss in console
        tqdm.write(
            f"[LOG] Epoch {i} Total Loss: {total_loss.item()}, Var Loss: {var_loss.item()}, Grad Loss: {grad_loss.item()},"
        )
        # update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_factor
        wandb_logger.write("learing_rate", param_group['lr'])

        # valid
        if (i+1) % 10 == 0:
            time_analyzer.start_valid()
            with torch.no_grad():
                valid_ref = warpper.get_reference_time(valid_norm_txy, "min")
                valid_warped_norm_txy = warpper.warp_events(valid_norm_txy, valid_ref, method=warp_config["solver"])
                # create image warped event 
                num_iwe_valid = valid_warped_norm_txy.shape[0]
                num_events_valid = valid_warped_norm_txy.shape[1]          
                valid_polarity = valid_events[:, 3].unsqueeze(0).expand(num_iwe_valid, num_events_valid).unsqueeze(-1).to(device)
                valid_warped_norm_xytp = torch.cat((valid_warped_norm_txy[..., [1,2,0]], valid_polarity), dim=2)
                
                valid_warped_events = event_proc.normalized_plane_to_pixel(
                    valid_warped_norm_xytp, 
                    camera_K
                )
                valid_warped_events = valid_warped_events[...,[1,0,2,3]]
            
                optimized_iwe = converter.create_iwes(
                    events=valid_warped_events,
                    method="bilinear_vote",
                    sigma=1.0,
                    blur=False
                )
                wandb_logger.write_img("iwe", optimized_iwe.detach().cpu().numpy() * 255)
            time_analyzer.end_valid()

        # log
        time_analyzer.end_epoch()
        wandb_logger.update_buffer()

    stats = time_analyzer.get_statistics()
    wandb_logger.write("Total Trainning Time", (stats["total_train_time"] / 60))
    wandb_logger.write("Average Epoch Time", (stats["avg_epoch_time"] / 60))
    wandb_logger.write("Total Validation Time", (stats["total_valid_time"] / 60))
    wandb_logger.write("Average Validation Time", (stats["avg_valid_time"] / 60))
    wandb_logger.update_buffer()
    
    # save model
    log_model_path = config["logger"]["model_weight_path"]
    dir_name = os.path.dirname(log_model_path) 
    os.makedirs(dir_name, exist_ok=True)
    torch.save(flow_field.state_dict(), log_model_path)



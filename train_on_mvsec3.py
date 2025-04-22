import os
import sys
import wandb
import torch 
import logging
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as mpl_fig
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from typing import Dict

from src.loss.focus import FocusLoss
from src.loss.motion import MotionLoss
from src.loss.dec import differential_epipolar_constrain
from src.utils import misc
from src.utils import pose
from src.utils import event_proc
from src.utils import flow_proc
from src.utils import metric
from src.utils import vector_math
from src.utils import load_config
from src.model.eventflow import EventFlowINR
from src.utils.wandb import WandbLogger
from src.model.warp import NeuralODEWarp, NeuralODEWarpV2
from src.model.geometric import Pixel2Cam, DiffEpipolarTheseusOptimizer, MotionModel
from src.utils.timer import TimeAnalyzer
from src.loader import dataset_manager
from src.model.eventflow import DenseOpticalFlowCalc
from src.utils.event_image_converter import EventImageConverter
from src.utils.visualizer import Visualizer

from src.utils.vector_math import vector_to_skew_matrix
from src.loader.MVSEC.loader import MVSECDataLoader
from src.utils.filter import MedianPool2d


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

torch.set_float32_matmul_precision('high')

def run_train_phase(
    args, 
    config: Dict, 
    dataset: MVSECDataLoader,
    # warpper: NeuralODEWarp,
    # nn_optimizer: torch.optim,
    criterion: Dict,
    imager: EventImageConverter,
    viz: Visualizer,
    wandb_logger: WandbLogger,
    time_analyzer: TimeAnalyzer,
    device: torch.device
):  
    # load gt velocity
    gt_lin_vel_array, gt_ang_vel_array = dataset.load_gt_motion()
    gt_lin_vel_all = torch.from_numpy(gt_lin_vel_array).clone().to(device)
    gt_ang_vel_all = torch.from_numpy(gt_ang_vel_array).clone().to(device)
    gt_lin_vel_spline = pose.create_vel_cspline(gt_lin_vel_all)
    gt_ang_vel_spline = pose.create_vel_cspline(gt_ang_vel_all)
    
    # generate coordinates on image plane
    intrinsic_mat = torch.from_numpy(dataset.intrinsic).clone()
    pixel2cam_converter = Pixel2Cam(
        dataset._HEIGHT, dataset._WIDTH, 
        intrinsic_mat,
        device
    )
    image_coords = pixel2cam_converter.generate_image_coordinate()
    norm_image_coords = pixel2cam_converter.generate_normalized_image_coordinate()
    
    # Split events stream 
    eval_frame_timestamp_list = dataset.eval_frame_time_list()
    eval_dt_list = [4]
    gt_dt = 4
    total_batch_events = []
    total_gt_flow = []
    
    for eval_dt in tqdm(eval_dt_list, desc=f"Split the events into batches for validation", leave=True):
        for i1 in tqdm(
            range(len(eval_frame_timestamp_list) - eval_dt), 
            desc=f"Divide the event stream into {eval_dt}-frame intervals", leave=True
        ):  
            t1 = eval_frame_timestamp_list[i1]
            t2 = eval_frame_timestamp_list[i1 + eval_dt]
            gt_t1 = eval_frame_timestamp_list[i1]
            gt_t2 = eval_frame_timestamp_list[i1 + gt_dt]
            
            gt_flow = dataset.load_optical_flow(gt_t1, gt_t2)
            ind1 = dataset.time_to_index(t1)  # event index
            ind2 = dataset.time_to_index(t2)
            batch_events = dataset.load_event(ind1, ind2)
            batch_events[...,2] -= dataset.min_ts # norm_t
            total_batch_events.append(batch_events)
            total_gt_flow.append(gt_flow)

    epe = []
    ae = []
    out = []
    
    for i in tqdm(range(len(total_batch_events))):
        valid_events_origin = torch.tensor(total_batch_events[i], dtype=torch.float32).to(device)
        valid_origin_iwe = viz.create_clipped_iwe_for_visualization(valid_events_origin)
        viz.visualize_image(valid_origin_iwe.squeeze(0), file_prefix="origin_iwe_" + str(i))
        
        gt_flow = total_gt_flow[i]
        gt_flow = np.transpose(gt_flow, (2, 0, 1))  # [2, H, W]
        viz.visualize_optical_flow(
            gt_flow[0],
            gt_flow[1],
            visualize_color_wheel=True,
            file_prefix="gt_flow" + str(i),
        )

        # reset model
        misc.fix_random_seed()
        flow_field = EventFlowINR(model_config).to(device)
        warpper = NeuralODEWarpV2(flow_field, device, **warp_config)
        dec_theseus_optimizer = DiffEpipolarTheseusOptimizer(image_size, device)
        optimizer = optim.Adam(warpper.flow_field.parameters(), lr=optimizer_config["initial_lrate"])
        
        # decay lr
        decay_steps = 1
        num_epochs = config["optimizer"]["num_epoch"]
        final_lrate = config["optimizer"]["final_lrate"]
        initial_lrate = config["optimizer"]["initial_lrate"]
        decay_factor = (final_lrate / initial_lrate) ** (1 / (num_epochs / decay_steps))
        # print(f"learning rate decay factor:{decay_factor}")
        # initial_lr = optimizer.param_groups[0]['lr']
        print(f"Segment {i}")
        iter = 0

        if(i!=1825): # 1825
            continue
        for j in tqdm(range(num_epochs)):
            time_analyzer.start_epoch()
            
            optimizer.zero_grad()
            
            batch_events = valid_events_origin.clone() # (y, x, t, p)
            origin_iwe = imager.create_iwes(batch_events)
            batch_events = batch_events[..., [2, 1, 0, 3]]  # (t, x, y, p)
            batch_events_txy = batch_events[:, :3]
            
            # get t_ref
            t_ref = warpper.get_reference_time(batch_events_txy, warp_config["tref_setting"])

            # warp by neural ode 
            warped_events_txy = warpper.warp_events(batch_events_txy, t_ref).unsqueeze(0)

            # create image warped event    
            num_iwe, num_events = warped_events_txy.shape[0], warped_events_txy.shape[1]
            polarity = batch_events[:, 3].unsqueeze(0).expand(num_iwe, num_events).unsqueeze(-1)
            warped_events = torch.cat((warped_events_txy[..., [1,2,0]], polarity), dim=2).to(device)
            warped_events = warped_events[...,[1,0,2,3]] # (x,y,t,p) ——> (y,x,t,p)
            
            # events should be reorganized as (y,x,t,p) that can processed by create_iwes
            # if(j < 500):
            #     blur_kernel_sigma = 3
            # else:
            #     blur_kernel_sigma = 1
            iwes = imager.create_iwes(
                events=warped_events,
                method="bilinear_vote",
                sigma=1,
                blur=True
            ) # [n,h,w] n can be one or multi
            
            # CMax loss
            if num_iwe == 1:
                var_loss = criterion["var_criterion"](iwes)
                grad_loss = criterion["grad_criterion"](iwes)
            elif num_iwe > 1:
                # var_loss = torch.Tensor([0]).to(device)
                grad_loss_t_min = grad_criterion(iwes[0].unsqueeze(0))
                grad_loss_t_mid = grad_criterion(iwes[len(iwes)//2].unsqueeze(0))
                grad_loss_t_max = grad_criterion(iwes[-1].unsqueeze(0))
                grad_loss_origin = grad_criterion(origin_iwe)
                grad_loss = (grad_loss_t_min + grad_loss_t_max + 2 * grad_loss_t_mid) / 4 * grad_loss_origin
                
                var_loss_t_min = var_criterion(iwes[0].unsqueeze(0))
                var_loss_t_mid = var_criterion(iwes[len(iwes)//2].unsqueeze(0))
                var_loss_t_max = var_criterion(iwes[-1].unsqueeze(0))
                var_loss_origin = var_criterion(origin_iwe)
                var_loss = (var_loss_t_min + var_loss_t_max + 2 * var_loss_t_mid) / 4 * var_loss_origin
            
            # sample coordinates
            events_mask = imager.create_eventmask(warped_events)
            events_mask = events_mask[0].squeeze(0).to(device)
            sample_coords = pixel2cam_converter.sample_sparse_coordinates(
                coord_tensor=image_coords,
                mask=events_mask,
                n=1000
            )
            t_ref_expanded = t_ref * torch.ones(sample_coords.shape[1], device=device).reshape(1,-1,1).to(device)
            sample_txy = torch.cat((t_ref_expanded, sample_coords[...,0:2]), dim=2)
            
            # get optical flow and coordinate on normalized plane
            sample_flow = warpper.flow_field.forward(sample_txy)
            sample_flow = F.pad(sample_flow, (0, 1), mode='constant', value=0)
            sample_norm_coords = flow_proc.pixel_to_normalized_coords(sample_coords, intrinsic_mat)
            sample_norm_flow = flow_proc.flow_to_normalized_coords(sample_flow, intrinsic_mat)
            
            # get gt velocity
            v_gt, w_gt = gt_lin_vel_spline.evaluate(t_ref), gt_ang_vel_spline.evaluate(t_ref)
            v_gt, w_gt = v_gt.to(torch.float32).unsqueeze(0), w_gt.to(torch.float32).unsqueeze(0)
            v_gt_norm = v_gt / torch.norm(v_gt, p=2, dim=-1, keepdim=True)
            
            w_gt = torch.tensor([0.0,0.0,0.0], dtype=torch.float32, device=device)
            dec_error, average_dec_loss = differential_epipolar_constrain(sample_norm_coords, sample_norm_flow, v_gt, w_gt)
            
            if(j % 100 == 0):
                hist_figure = metric.analyze_error_histogram(dec_error)
                if(isinstance(hist_figure, mpl_fig.Figure)):
                    wandb_logger.write_img("error_distribution", hist_figure)
                    hist_figure_save_dir = config["logger"]["results_dir"]
                    hist_figure_save_path = os.path.join(hist_figure_save_dir, f"error_distribution_{j}.png")
                    hist_figure.savefig(hist_figure_save_path, dpi=300, bbox_inches="tight")
                    plt.close(hist_figure)

            # intial value for differential epipolar constrain
            v_noise_level = 0.01
            # w_noise_level = 0.01
            v_gt_noisy = v_gt + v_noise_level * torch.randn_like(v_gt, device=v_gt.device)
            # w_gt_noisy = w_gt + w_noise_level * torch.randn_like(w_gt, device=w_gt.device)
            v_gt_noisy = v_gt_noisy / torch.norm(v_gt_noisy, p=2, dim=-1, keepdim=True)
            # init_velc = [v_gt_noisy, w_gt_noisy]
            init_velc = [v_gt_noisy]
            
            # theseus optimize 
            v_opt, v_his, err_his = dec_theseus_optimizer.optimize(
                sample_norm_coords, sample_norm_flow,  
                motion_model=MotionModel.PURE_TRANSLATION,
                init_velc=init_velc,
                num_iterations=100
            )

            if(j>1000):
                theseus_result = {
                    "v_his":v_his[:20],
                    "v_init": v_gt_noisy,
                    "v_opt": v_opt,
                    "v_gt": v_gt_norm,
                    # "w_his": w_his[:20],
                    # "w_init": w_gt_noisy,
                    # "w_opt": w_opt,
                    "w_gt": w_gt, 
                    "err_his": err_his[:20]
                }
                misc.save_theseus_result_as_text(theseus_result, config["logger"]["results_dir"])

            v_dir_error = vector_math.vector_dir_error_in_degrees(v_opt, v_gt_norm)
            # w_dir_error = vector_math.vector_dir_error_in_degrees(w_opt, w_gt)
            v_mag_error = vector_math.vector_mag_error(v_opt, v_gt_norm)
            # w_mag_error = vector_math.vector_mag_error(w_opt, w_gt)
            
            # Motion loss
            w_opt = torch.tensor([0.0,0.0,0.0], dtype=torch.float32, device=device)
            ssl_dec_error, ssl_average_dec_loss = differential_epipolar_constrain(sample_norm_coords, sample_norm_flow, v_opt, w_opt)
                    
            # Total loss
            alpha, beta, gamma = 1, 1, 5
            scaled_grad_loss = alpha * grad_loss
            scaled_var_loss = beta * var_loss
            scaled_ssl_dec_loss = 180 * ssl_average_dec_loss
            
            # if(j<500):
            #     total_loss = - (scaled_grad_loss + scaled_var_loss)
            # else:
            #     total_loss = - (scaled_grad_loss + scaled_var_loss) + scaled_motion_loss
            
            # if(j<200):
            #     total_loss = (1.0 / scaled_grad_loss)
            # else:
            #     total_loss = (1.0 / scaled_grad_loss) + scaled_motion_loss
            
            if(j<200):
                total_loss = (1.0 / (scaled_grad_loss + scaled_var_loss)) 
            else:
                total_loss = (1.0 / (scaled_grad_loss + scaled_var_loss)) + scaled_ssl_dec_loss
            # total_loss = (1.0 / (scaled_grad_loss + scaled_var_loss)) 
            
            # step
            total_loss.backward()
            optimizer.step()
            
            # update lr
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_factor
            iter+=1  
            
            wandb_logger.write("var_loss", scaled_var_loss.item())
            wandb_logger.write("grad_loss", scaled_grad_loss.item())
            wandb_logger.write("dec_loss", scaled_ssl_dec_loss.item())
            wandb_logger.write("train_loss", total_loss.item())
            wandb_logger.write("v_dir_error", v_dir_error.item())
            # wandb_logger.write("w_dir_error", w_dir_error.item())
            wandb_logger.write("v_mag_error", v_mag_error.item())
            # wandb_logger.write("w_mag_error", w_mag_error.item())
            wandb_logger.update_buffer()
        
            # valid
            if(j % 100 == 0):
                time_analyzer.start_valid()
                with torch.no_grad():
                    valid_events = valid_events_origin.clone()
                    valid_events = valid_events[:, [2, 1, 0, 3]] # (y, x, t, p) ——> (t, x, y, p)
                    valid_events_txy = valid_events[:, :3]
                    
                    valid_ref = warpper.get_reference_time(valid_events_txy, "max")
                    warped_valid_events_txy = warpper.warp_events(valid_events_txy, valid_ref).unsqueeze(0)
                    
                    # create image warped event 
                    num_iwe_valid, num_events_valid = warped_valid_events_txy.shape[0], warped_valid_events_txy.shape[1]       
                    valid_polarity = valid_events[:, 3].unsqueeze(0).expand(num_iwe_valid, num_events_valid).unsqueeze(-1).to(device)
                    warped_valid_events = torch.cat((warped_valid_events_txy[..., [1,2,0]], valid_polarity), dim=2).to(device)
                    warped_valid_events = warped_valid_events[...,[1,0,2,3]]
                    
                    # log
                    valid_warped_iwe = viz.create_clipped_iwe_for_visualization(warped_valid_events)
                    viz.visualize_image(valid_warped_iwe.squeeze(0), file_prefix="pred_iwe_"+ str(i))
                    
                    flow_calculator = DenseOpticalFlowCalc(
                        grid_size=image_size, 
                        intrinsic_mat = intrinsic_mat,
                        model=warpper.flow_field, 
                        normalize_coords_mode="None",
                        device=device
                    )
                    t_start = eval_frame_timestamp_list[i] - dataset.min_ts
                    t_end = eval_frame_timestamp_list[i + gt_dt] - dataset.min_ts
                    duration  = t_end - t_start
                    
                    pred_flow = flow_calculator.integrate_flow(t_start, t_end) # [H*W , 2]
                    pred_flow = pred_flow.cpu().numpy()
                    
                    viz.visualize_optical_flow_on_event_mask(pred_flow, valid_events_origin.cpu().numpy(), file_prefix="pred_flow_masked" + str(i))
                    
                    pred_flow = torch.from_numpy(pred_flow).unsqueeze(0).to(device)
                    gt_flow_tensor = torch.from_numpy(gt_flow).unsqueeze(0).to(device)
                    event_mask = imager.create_eventmask(valid_events_origin).to(device)
                    flow_error, _ = metric.calculate_flow_error(gt_flow_tensor, pred_flow, event_mask=event_mask)  # type: ignore
                    
                    epe.append(flow_error["EPE"].item())
                    ae.append(flow_error["AE"].item())
                    out.append(flow_error["3PE"].item())
                    wandb_logger.write("EPE", flow_error["EPE"].item())
                    wandb_logger.write("AE", flow_error["AE"].item())
                    wandb_logger.write("1PE", flow_error["1PE"].item())
                    wandb_logger.write("2PE", flow_error["2PE"].item())
                    wandb_logger.write("3PE", flow_error["3PE"].item())
                    wandb_logger.update_buffer()
                
                time_analyzer.end_valid()
                time_analyzer.end_epoch()
    error_dict = {"EPE": np.mean(epe), "AE": np.mean(ae), "3PE": np.mean(out)}
    logger.info(f"Average EPE: {np.mean(epe)}")
    logger.info(f"Average AE: {np.mean(ae)}")
    logger.info(f"Average 3PE: {np.mean(out)}")
    misc.save_flow_error_as_text(error_dict, config["logger"]["results_dir"])
    stats = time_analyzer.get_statistics()
    return warpper.flow_field, stats

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
    dataloader_config = data_config.pop('loader')
    dataset = MVSECDataLoader(config=data_config)
    dataset.set_sequence(data_config["sequence"])

    # event2img converter
    image_size = (data_config["hight"], data_config["width"])
    imager = EventImageConverter(image_size)

    # Visualizer
    viz = Visualizer(
        image_shape=image_size,
        show=False,
        save=True,
        save_dir=config["logger"]["results_dir"],
    )
    
    # instantiate criterion
    grad_criterion = FocusLoss(loss_type="gradient_magnitude", norm="l1")
    var_criterion = FocusLoss(loss_type="variance", norm="l1")
    motion_criterion = MotionLoss(loss_type="MSE")
    criterions = {
        "grad_criterion": grad_criterion, 
        "var_criterion": var_criterion,
        "motion_criterion": motion_criterion
    } 

    # time analysis
    time_analyzer = TimeAnalyzer()

    trained_flow_field, time_stats = run_train_phase(
        args, config, dataset, criterions,
        imager, viz, wandb_logger, time_analyzer, device
    )
    
    wandb_logger.write("Total Training Time (min)", time_stats["total_train_time"] / 60)
    wandb_logger.write("Average Epoch Time (min)", time_stats["avg_epoch_time"] / 60)
    wandb_logger.write("Total Validation Time (min)", time_stats["total_valid_time"] / 60)
    wandb_logger.write("Average Validation Time (min)", time_stats["avg_valid_time"] / 60)
    wandb_logger.update_buffer()
    
    # Save model
    log_model_path = config["logger"]["model_weight_path"]
    dir_name = os.path.dirname(log_model_path) 
    os.makedirs(dir_name, exist_ok=True)
    torch.save(trained_flow_field.state_dict(), log_model_path)
    
    wandb_logger.finish()
    
    
    


    
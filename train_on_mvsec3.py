import os
import sys
import torch 
import logging
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Dict
from dataclasses import dataclass

from src.utils import (
    misc,
    pose,
    metric,
    flow_proc,
    event_proc,
    load_config
)

from src.loader.MVSEC.loader import MVSECDataLoader

from src.loss.focus import FocusLoss
from src.loss.dec import DifferentialEpipolarLoss

from src.model.warp import NeuralODEWarpV2
from src.model.eventflow import EventFlowINR, DenseFlowExtractor
from src.model.geometric import Pixel2Cam, CubicBsplineVelocityModel
from src.model.scheduler import create_exponential_scheduler

from src.utils.wandb import WandbLogger
from src.utils.timer import TimeAnalyzer
from src.utils.visualizer import Visualizer
from src.utils.event_imager import EventImager

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

@dataclass
class Tools:
    viz: Visualizer
    imager: EventImager
    # wandb_logger: WandbLogger
    time_analyzer: TimeAnalyzer
    
def split_events(config: Dict, dataset: MVSECDataLoader, viz: Visualizer):
    eval_dt = config["data"]["eval_dt"]
    eval_frame_timestamp_list = dataset.eval_frame_time_list() 
    if config["data"]["sequence"] == "outdoor_day1":
        eval_frame_timestamp_list = eval_frame_timestamp_list[9910:10710]
    if misc.check_key_and_bool(config["data"], "remove_car"):
        logger.info("Remove car-boody pixels")
    
    total_batch_events = []
    total_batch_gt_flow = []
    gt_flow_save_dir = os.path.join(config["logger"]["results_dir"], "gt_flow")
    origin_iwe_save_dir = os.path.join(config["logger"]["results_dir"], "origin_iwe")
    
    for i1 in tqdm(
        range(len(eval_frame_timestamp_list) - eval_dt), 
        desc=f"Divide the event stream into {eval_dt}-frame intervals", leave=True
    ):  
        t1 = eval_frame_timestamp_list[i1]
        t2 = eval_frame_timestamp_list[i1 + eval_dt]
    
        ind1, ind2 = dataset.time_to_index(t1), dataset.time_to_index(t2)
        batch_events = dataset.load_event(ind1, ind2)
        if misc.check_key_and_bool(config["data"], "remove_car"):
            batch_events = event_proc.crop_event(batch_events, 0, 193, 0, 346)
        batch_events[...,2] -= dataset.min_ts   # norm_t
        gt_flow = dataset.load_optical_flow(t1, t2)
        
        batch_events_tensor = torch.tensor(batch_events, dtype=torch.float32).to(device)
        batch_events_iwe = viz.create_clipped_iwe_for_visualization(batch_events_tensor)
        viz.update_save_dir(origin_iwe_save_dir)
        viz.visualize_image(batch_events_iwe.squeeze(0), file_prefix="origin_iwe_")
        
        gt_flow = np.transpose(gt_flow, (2, 0, 1))  # [2, H, W]
        viz.update_save_dir(gt_flow_save_dir)
        viz.visualize_optical_flow(
            gt_flow[0],
            gt_flow[1],
            visualize_color_wheel=True,
            file_prefix="gt_flow_"
        )
        
        total_batch_events.append(batch_events)
        total_batch_gt_flow.append(gt_flow)
        
    return total_batch_events, total_batch_gt_flow

def run_valid_phase(
    config: Dict, segment_index: int, events_origin: torch.Tensor,
    gt_flow: torch.Tensor, gt_motion: Dict,
    warpper: NeuralODEWarpV2,
    motion_spline: CubicBsplineVelocityModel,
    flow_calculator: DenseFlowExtractor,
    tools: Tools,
    device: torch.device
):
    pred_iwe_save_dir = os.path.join(config["logger"]["results_dir"], "pred_iwe")
    pred_flow_save_dir = os.path.join(config["logger"]["results_dir"], "pred_flow")
    motion_save_dir = os.path.join(config["logger"]["results_dir"], "motion")
    if not os.path.exists(motion_save_dir):
        os.makedirs(motion_save_dir)
        
    tools.time_analyzer.start_valid()
    with torch.no_grad():
        # warp events by trained flow
        valid_events = events_origin.clone()
        valid_events_txy = valid_events[:, [2, 1, 0, 3]][:, :3] # (y, x, t, p) ——> (t, x, y, p) ——> (t, x, y)
        valid_ref = warpper.get_reference_time(valid_events_txy, "max")
        warped_valid_events_txy = warpper.warp_events(valid_events_txy, valid_ref).unsqueeze(0)
        
        # create image warped event 
        num_iwe_valid, num_events_valid = warped_valid_events_txy.shape[0], warped_valid_events_txy.shape[1]       
        valid_polarity = valid_events[:, 3].unsqueeze(0).expand(num_iwe_valid, num_events_valid).unsqueeze(-1).to(device)
        warped_valid_events = torch.cat((warped_valid_events_txy[..., [2,1,0]], valid_polarity), dim=2).to(device)
        
        # save image warped event 
        valid_warped_iwe = tools.viz.create_clipped_iwe_for_visualization(warped_valid_events)
        tools.viz.update_save_dir(pred_iwe_save_dir)
        tools.viz.visualize_image(valid_warped_iwe.squeeze(0), file_prefix="pred_iwe_")
        
        # calculate optical flow error
        t_start = torch.min(valid_events_txy[:, 0])
        t_end = torch.max(valid_events_txy[:, 0])
        pred_flow = flow_calculator.integrate_flow(warpper.flow_field, t_start, t_end).unsqueeze(0) # [B,2,H,W]
        gt_flow = gt_flow.unsqueeze(0).to(device) # [B,2,H,W]
        event_mask = tools.imager.create_eventmask(valid_events).to(device)
        flow_error, _ = metric.calculate_flow_error(gt_flow, pred_flow, event_mask=event_mask)  # type: ignore
        
        # visualize dense optical flow
        pred_flow = pred_flow.squeeze(0).cpu().numpy()
        tools.viz.update_save_dir(pred_flow_save_dir)
        tools.viz.visualize_optical_flow_on_event_mask(pred_flow, valid_events.cpu().numpy(), file_prefix="pred_flow_masked_")
        
        # calculate motion error
        gt_lin_vel_spline, gt_ang_vel_spline = gt_motion["linear_velocity"], gt_motion["angular_velocity"]
        t_eval = torch.linspace(start=t_start, end=t_end, steps=101).to(device)
        gt_lin_vel, gt_ang_vel = gt_lin_vel_spline.evaluate(t_eval), gt_ang_vel_spline.evaluate(t_eval)
        t_eval_norm = torch.linspace(start=0, end=1, steps=101).to(device)
        eval_lin_vel, eval_ang_vel = motion_spline.forward(t_eval_norm)
        mag_gt_lin_vel = torch.norm(gt_lin_vel, p=2, dim=1, keepdim=True)
        eval_lin_vel = eval_lin_vel.abs() * mag_gt_lin_vel * gt_lin_vel.sign()
        eval_t_lin_vel = torch.cat([t_eval[50].reshape(1,1), eval_lin_vel[50].reshape(1,3)], dim=1)
        eval_t_ang_vel = torch.cat([t_eval[50].reshape(1,1), eval_ang_vel[50].reshape(1,3)], dim=1)
        eval_motion = {"linear_velocity": eval_t_lin_vel, "angular_velocity": eval_t_ang_vel}
        
        # plot velocity        
        fig_lin, fig_ang = misc.visualize_velocities(config, gt_lin_vel, gt_ang_vel, eval_lin_vel, eval_ang_vel, t_eval)
        lin_vel_figure_save_path = os.path.join(motion_save_dir, f"linear_velocity_comparison_{str(segment_index)}.png")
        ang_vel_figure_save_path = os.path.join(motion_save_dir, f"angular_velocity_comparison_{str(segment_index)}.png")
        fig_lin.savefig(lin_vel_figure_save_path, dpi=300, bbox_inches="tight")
        fig_ang.savefig(ang_vel_figure_save_path, dpi=300, bbox_inches="tight")
        plt.close(fig_lin)
        plt.close(fig_ang)
        
    tools.time_analyzer.end_valid()
    
    return flow_error, eval_motion
    
def run_train_phase(
    config: Dict, 
    dataset: MVSECDataLoader,
    criterion: Dict,
    tools: Tools,
    device: torch.device
):  
    model_config = config["model"]
    warp_config = config["warp"]
    loss_config = config["loss"]
    optimizer_config = config["optimizer"]
    
    # get gt motion spline
    gt_lin_vel_array, gt_ang_vel_array = dataset.load_gt_motion()
    gt_lin_vel_tensor = torch.from_numpy(gt_lin_vel_array).clone().to(device)
    gt_ang_vel_tensor = torch.from_numpy(gt_ang_vel_array).clone().to(device)
    gt_lin_vel_spline = pose.create_vel_cspline(gt_lin_vel_tensor)
    gt_ang_vel_spline = pose.create_vel_cspline(gt_ang_vel_tensor)
    gt_motion = {
        "linear_velocity": gt_lin_vel_spline,
        "angular_velocity": gt_ang_vel_spline
    }
    
    # generate coordinates on image plane
    intrinsic_mat = torch.from_numpy(dataset.intrinsic).clone()
    if misc.check_key_and_bool(config["data"], "remove_car"):
        logger.info("Correct intrinsic matrix")
        intrinsic_mat[1, 2] -= 0.5 * (260 - 193)
    pixel2cam_converter = Pixel2Cam(
        dataset._HEIGHT, dataset._WIDTH, 
        intrinsic_mat,
        device
    )
    image_coords = pixel2cam_converter.generate_image_coordinate()
    flow_calculator = DenseFlowExtractor(grid_size=image_size, device=device)
    
    # split events
    total_batch_events, total_batch_gt_flow = split_events(config=config, dataset=dataset, viz=tools.viz)
    # total_batch_events, total_batch_gt_flow = total_batch_events[397:412], total_batch_gt_flow[397:412]
    
    epe = []
    ae = []
    out =[]
    all_eval_lin_vel = torch.zeros(len(total_batch_events), 4, device=device)    # 形状 (N, 4)
    all_eval_ang_vel = torch.zeros(len(total_batch_events), 4, device=device)    # 形状 (N, 4)
    for i in tqdm(range(len(total_batch_events))):
        # reset model
        flow_field = EventFlowINR(model_config).to(device)
        warpper = NeuralODEWarpV2(flow_field, device, warp_config)
        motion_spline = CubicBsplineVelocityModel().to(device)
        nn_optimizer = optim.Adam(warpper.flow_field.parameters())
        spline_optimizer = optim.Adam(motion_spline.parameters(), lr=optimizer_config["spline_lr"])
        
        # learning rate scheduler
        num_iters = optimizer_config["num_iters"]
        nn_scheduler = create_exponential_scheduler(
            optimizer=nn_optimizer,
            lr1=optimizer_config["nn_initial_lr"],
            lr2=optimizer_config["nn_final_lr"],
            total_steps=num_iters
        )
        
        # segment i
        events_origin = torch.tensor(total_batch_events[i], dtype=torch.float32).to(device)
        gt_flow = torch.tensor(total_batch_gt_flow[i], dtype=torch.float32).to(device)
        print(f"Segment {i}")
        
        iter = 0
        tools.time_analyzer.start_epoch()
        for j in tqdm(range(num_iters)): 
            # reset optimizer
            nn_optimizer.zero_grad()
            spline_optimizer.zero_grad()
            
            # warp events by neural ode
            batch_events = events_origin.clone() # (y, x, t, p)
            batch_events_txy = batch_events[..., [2, 1, 0, 3]][:, :3] # (y, x, t, p) ——> (t, x, y, p) ——> (t, x, y)
            t_ref = warpper.get_reference_time(batch_events_txy, warp_config["tref_setting"])
            warped_events_txy = warpper.warp_events(batch_events_txy, t_ref).unsqueeze(0)

            # create image warped event    
            num_iwe, num_events = warped_events_txy.shape[0], warped_events_txy.shape[1]
            polarity = batch_events[:, 3].unsqueeze(0).expand(num_iwe, num_events).unsqueeze(-1)
            warped_events = torch.cat((warped_events_txy[..., [2,1,0]], polarity), dim=2).to(device) # (t, x, y) ——> (y,x,t,p)
            
            # events should be reorganized as (y,x,t,p) that can processed by create_iwes
            iwes = tools.imager.create_iwes(
                events=warped_events,
                method="bilinear_vote",
                sigma=1,
                blur=True
            ) # [n,h,w] n should be one
            
            grad_loss = 0
            var_loss = 0
            ssl_average_dec_loss = 0
            
            # CMax loss
            if misc.check_key_and_bool(loss_config, "cmax"):
                if num_iwe == 1:
                    var_loss = criterion["var_criterion"](iwes)
                    grad_loss = criterion["grad_criterion"](iwes)
                elif num_iwe > 1:
                    raise ValueError("Multiple iwes are not supported")
            
            # DEC loss
            if misc.check_key_and_bool(loss_config, "dec"):
                # sample coordinates
                events_mask = tools.imager.create_eventmask(warped_events)
                events_mask = events_mask[0].squeeze(0).to(device)
                sample_coords = pixel2cam_converter.sample_sparse_coordinates(
                    coord_tensor=image_coords,
                    mask=events_mask,
                    n=10000
                )
                t_ref_expanded = t_ref * torch.ones(sample_coords.shape[1], device=device).reshape(1,-1,1).to(device)
                sample_txy = torch.cat((t_ref_expanded, sample_coords[...,0:2]), dim=2)
                
                # get optical flow and coordinate on normalized plane
                sample_flow = warpper.flow_field.forward(sample_txy)
                sample_flow = torch.nn.functional.pad(sample_flow, (0, 1), mode='constant', value=0)
                sample_norm_coords = flow_proc.pixel_to_normalized_coords(sample_coords, intrinsic_mat)
                sample_norm_flow = flow_proc.flow_to_normalized_coords(sample_flow, intrinsic_mat)
                
                # differential epipolar constrain
                t_min, t_max = torch.min(batch_events_txy[..., 0]), torch.max(batch_events_txy[..., 0])
                interp_t = (t_ref - t_min) / (t_max - t_min)
                lin_vel, ang_vel = motion_spline.forward(interp_t.unsqueeze(0))
                ssl_dec_error, ssl_average_dec_loss = criterion["dec_criterion"](
                    sample_norm_coords, sample_norm_flow, 
                    lin_vel, ang_vel
                )
            # v_gt, w_gt = gt_lin_vel_spline.evaluate(t_ref), gt_ang_vel_spline.evaluate(t_ref)
            # v_gt, w_gt = v_gt.to(torch.float32).unsqueeze(0), w_gt.to(torch.float32).unsqueeze(0)
            # gt_dec_error, gt_average_dec_loss = differential_epipolar_constrain(
            #     sample_norm_coords, sample_norm_flow, 
            #     v_gt, w_gt
            # )
                    
            # Total loss
            alpha, beta, gamma = 1, 1, 2.5   # 2.5
            scaled_grad_loss = alpha * grad_loss
            scaled_var_loss = beta * var_loss
            scaled_ssl_dec_loss = gamma * ssl_average_dec_loss
            
            if(j<=200):
                total_loss = - (scaled_grad_loss + scaled_var_loss)
            else:
                total_loss = - (scaled_grad_loss + scaled_var_loss) + scaled_ssl_dec_loss
            
            # step
            total_loss.backward()
            nn_optimizer.step()
            nn_scheduler.step()
            spline_optimizer.step()
            iter+=1  
            
        tools.time_analyzer.end_epoch()
        
        flow_error, eval_motion = run_valid_phase(
            config, i, events_origin, gt_flow, gt_motion, 
            warpper, motion_spline,flow_calculator, tools, device
        )
        epe.append(flow_error["EPE"])
        ae.append(flow_error["AE"])
        out.append(flow_error["3PE"])
        
        all_eval_lin_vel[i] = eval_motion["linear_velocity"]
        all_eval_ang_vel[i] = eval_motion["angular_velocity"]
    
    all_eval_lin_vel_spline = pose.create_vel_cspline(all_eval_lin_vel)
    all_eval_ang_vel_spline = pose.create_vel_cspline(all_eval_ang_vel)
    all_motion_eval_t = torch.from_numpy(gt_lin_vel_array[:, 0]).to(device)
    eval_lin_vel_tensor = all_eval_lin_vel_spline.evaluate(all_motion_eval_t)
    eval_ang_vel_tensor = all_eval_ang_vel_spline.evaluate(all_motion_eval_t)
    
    fig_lin, fig_ang = misc.visualize_velocities(
        gt_lin_vel_tensor[:, 1:], gt_ang_vel_tensor[:, 1:], 
        eval_lin_vel_tensor, eval_ang_vel_tensor, 
        all_motion_eval_t
    )
    motion_save_dir = os.path.join(config["logger"]["results_dir"], "motion")
    lin_vel_figure_save_path = os.path.join(motion_save_dir, f"linear_velocity_comparison_whole_seq.png")
    ang_vel_figure_save_path = os.path.join(motion_save_dir, f"angular_velocity_comparison_whole_seq.png")
    fig_lin.savefig(lin_vel_figure_save_path, dpi=300, bbox_inches="tight")
    fig_ang.savefig(ang_vel_figure_save_path, dpi=300, bbox_inches="tight")
    plt.close(fig_lin)
    plt.close(fig_ang)
    
    rmse_lin, rmse_ang = metric.calculate_motion_error(
        eval_lin_vel_tensor, eval_ang_vel_tensor,
        gt_lin_vel_tensor[:, 1:], gt_ang_vel_tensor[:, 1:]
    )
    
    error_dict = {
        "EPE": np.mean(epe), 
        "AE": np.mean(ae), 
        "3PE": np.mean(out),
        "RMSE_linear": rmse_lin,
        "RMSE_angular": rmse_ang
    }
    
    logger.info(f"Average EPE: {np.mean(epe)}")
    logger.info(f"Average AE: {np.mean(ae)}")
    logger.info(f"Average 3PE: {np.mean(out)}")
    logger.info(f"RMSE_linear: {rmse_lin}")
    logger.info(f"RMSE_angular: {rmse_ang}")
    misc.save_flow_error_as_text(error_dict, config["logger"]["results_dir"])
    stats = tools.time_analyzer.get_statistics()
    return warpper.flow_field, stats

if __name__ == "__main__":
    # load configs
    args = load_config.parse_args()
    config = load_config.load_yaml_config(args.config)
    
    # detect cuda
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    logger.info(f"Use device: {device}")
    
    # fix seed
    misc.fix_random_seed()

    # load dataset 
    dataloader_config = config["data"].pop('loader')
    dataset = MVSECDataLoader(config=config["data"])
    dataset.set_sequence(config["data"]["sequence"])

    # instantiate criterion
    dec_criterion = DifferentialEpipolarLoss()
    var_criterion = FocusLoss(loss_type="variance", norm="l1")
    grad_criterion = FocusLoss(loss_type="gradient_magnitude", norm="l1")
    criterions = {
        "grad_criterion": grad_criterion, 
        "var_criterion": var_criterion,
        "dec_criterion": dec_criterion
    } 

    # wandb
    # wandb_logger = WandbLogger(config)
    
    # event2img converter
    image_size = (config["data"]["hight"], config["data"]["width"])
    imager = EventImager(image_size)

    # Visualizer
    viz = Visualizer(
        image_shape=image_size,
        show=False,
        save=True,
        save_dir=config["logger"]["results_dir"],
    )

    # time analysis
    time_analyzer = TimeAnalyzer()

    # create tools
    tools = Tools(
        viz=viz,
        imager=imager,
        # wandb_logger=wandb_logger,
        time_analyzer=time_analyzer
    )
    
    # trainer
    trained_flow_field, time_stats = run_train_phase(config, dataset, criterions, tools, device)
    
    logger.info("Total Training Time (min)", time_stats["total_train_time"])
    logger.info("Average Epoch Time (min)", time_stats["avg_epoch_time"])
    logger.info("Average Epoch Time (min)", time_stats["avg_epoch_time"])
    logger.info("Total Validation Time (min)", time_stats["total_valid_time"])
    logger.info("Average Validation Time (min)", time_stats["avg_valid_time"])
    
    misc.save_time_log_as_text(time_stats, config["logger"]["results_dir"])
    
    # Save model
    log_model_path = config["logger"]["model_weight_path"]
    dir_name = os.path.dirname(log_model_path) 
    os.makedirs(dir_name, exist_ok=True)
    torch.save(trained_flow_field.state_dict(), log_model_path)
    
    # tools.wandb_logger.finish()
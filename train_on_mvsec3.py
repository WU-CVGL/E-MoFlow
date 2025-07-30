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
from src.model.early_stopping import EarlyStopping, EarlyStoppingMode, EarlyStoppingStats
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
    data_config = config["data"]
    eval_dt = data_config["eval_dt"]
    n_events = data_config["n_events_per_batch"]
    eval_frame_timestamp_list = dataset.eval_frame_time_list() 
    if data_config["sequence"] in ["outdoor_day1", "outdoor_day2"]:
        eval_frame_timestamp_list = eval_frame_timestamp_list[
            data_config["valid_frame_1"] : data_config["valid_frame_2"]
        ]
    if misc.check_key_and_bool(data_config, "remove_car"):
        logger.info("Remove car-boody pixels")
    
    total_batch_events = []
    total_batch_events_for_train = []
    total_batch_gt_flow = []
    gt_flow_save_dir = os.path.join(config["logger"]["results_dir"], "gt_flow")
    origin_iwe_save_dir = os.path.join(config["logger"]["results_dir"], "origin_iwe")
    txt_save_dir = os.path.join(config["logger"]["results_dir"], "output.txt")
    
    for i1 in tqdm(
        range(len(eval_frame_timestamp_list) - eval_dt), 
        desc=f"Divide the event stream into {eval_dt}-frame intervals", leave=True
    ):  
        # if(i1!=419):
        #     continue
        # else:
            
        t1 = eval_frame_timestamp_list[i1]
        t2 = eval_frame_timestamp_list[i1 + eval_dt]
    
        ind1, ind2 = dataset.time_to_index(t1), dataset.time_to_index(t2)
        batch_events = dataset.load_event(ind1, ind2)
        batch_events[..., 2] -= dataset.min_ts
        gt_flow = dataset.load_optical_flow(t1, t2)
        if ind2 - ind1 < n_events:
            insufficient = n_events - (ind2 - ind1)
            ind1 -= insufficient // 2
            ind2 += insufficient // 2
        elif ind2 - ind1 > n_events:
            ind1 = ind2 - n_events
        
        batch_events_for_train = dataset.load_event(max(ind1, 0), min(ind2, len(dataset)))
        batch_events_for_train[..., 2] -= dataset.min_ts
        if misc.check_key_and_bool(data_config, "remove_car"):
            batch_events_for_train = event_proc.crop_event(batch_events_for_train, 0, 193, 0, 346)
            batch_events = event_proc.crop_event(batch_events, 0, 193, 0, 346)
            
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
        # fmt = ["%d", "%d", "%f", "%d"]
        # np.savetxt(txt_save_dir, batch_events_for_train, fmt=fmt, delimiter=" ")
        
        total_batch_events.append(batch_events)
        total_batch_events_for_train.append(batch_events_for_train)
        total_batch_gt_flow.append(gt_flow)
        
    return total_batch_events, total_batch_events_for_train, total_batch_gt_flow

def run_valid_phase(
    config: Dict, segment_index: int, batch_events: torch.Tensor,
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
        valid_events = batch_events.clone()
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
        flow_metric, _ = metric.calculate_flow_metric(gt_flow, pred_flow, event_mask=event_mask)  # type: ignore
        
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
        rmse_lin, rmse_ang, e_lin, e_ang = metric.calculate_motion_metric(
            eval_lin_vel, eval_ang_vel, gt_lin_vel, gt_ang_vel
        )
        motion_metric = {
            "rmse_linear": rmse_lin, 
            "rmse_angular": rmse_ang,
            "error_linear": e_lin,
            "error_angular": e_ang
        }
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
    
    return flow_metric, motion_metric, eval_motion
    
def run_train_phase(
    config: Dict, 
    dataset: MVSECDataLoader,
    criterion: Dict,
    tools: Tools,
    device: torch.device
):  
    data_config = config["data"]
    model_config = config["model"]
    warp_config = config["warp"]
    loss_config = config["loss"]
    optimizer_config = config["optimizer"]
    early_stopping_config = config.get("early_stopping", {})
    use_early_stopping = early_stopping_config.get("enabled", False)
    
    # stats of early stopping
    early_stopping_stats = EarlyStoppingStats(
        segment_iterations=[],
        early_stopped_segments=[],
        total_segments=0
    )
    
    # split events
    total_batch_events, total_batch_events_for_train, total_batch_gt_flow = split_events(config=config, dataset=dataset, viz=tools.viz)
    
    # number of total segments
    early_stopping_stats.total_segments = len(total_batch_events)
    
    # get gt motion spline
    gt_lin_vel_array, gt_ang_vel_array = dataset.load_gt_motion()
    if if data_config["sequence"] in ["outdoor_day1", "outdoor_day2"]:
        valid_t_min = np.min(total_batch_events[0][:, 2])
        valid_t_max = np.max(total_batch_events[-1][:, 2])
        gt_lin_vel_array = gt_lin_vel_array[
            (gt_lin_vel_array[:, 0] >= valid_t_min) & (gt_lin_vel_array[:, 0] <= valid_t_max)
        ]
        gt_ang_vel_array = gt_ang_vel_array[
            (gt_ang_vel_array[:, 0] >= valid_t_min) & (gt_ang_vel_array[:, 0] <= valid_t_max)
        ]
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
    if misc.check_key_and_bool(data_config, "remove_car"):
        logger.info("Correct intrinsic matrix")
        intrinsic_mat[1, 2] -= 0.5 * (260 - 193)
    pixel2cam_converter = Pixel2Cam(
        dataset._HEIGHT, dataset._WIDTH, 
        intrinsic_mat,
        device
    )
    image_coords = pixel2cam_converter.generate_image_coordinate()
    flow_calculator = DenseFlowExtractor(grid_size=image_size, device=device, warp_config=warp_config)
    
    epe, ae, out = [], [], []
    rmse_lin, rmse_ang, e_lin, e_ang = [], [], [], []
    all_eval_lin_vel = torch.zeros(len(total_batch_events), 4, device=device) # shape:(N, 4)   
    all_eval_ang_vel = torch.zeros(len(total_batch_events), 4, device=device) # shape:(N, 4)
    # motion_spline = CubicBsplineVelocityModel().to(device)
    # flow_field = EventFlowINR(model_config).to(device)
    for i in tqdm(range(len(total_batch_events))):
        # reset model
        flow_field = EventFlowINR(model_config).to(device)
        warpper = NeuralODEWarpV2(flow_field, device, warp_config)
        motion_spline = CubicBsplineVelocityModel().to(device)
        nn_optimizer = optim.Adam(warpper.flow_field.parameters())
        spline_optimizer = optim.Adam(motion_spline.parameters(), lr=optimizer_config["spline_lr"])
        
        use_early_stopping = early_stopping_config["enabled"]
        if use_early_stopping:
            early_stopping = EarlyStopping(
                burn_in_steps=early_stopping_config.get("burn_in_steps", 100),
                mode=EarlyStoppingMode.MIN,
                min_delta=early_stopping_config.get("min_delta", 1e-6),
                patience=early_stopping_config.get("patience", 50),
                percentage=False
            )
        
        # learning rate scheduler
        num_iters = optimizer_config["num_iters"]
        nn_scheduler = create_exponential_scheduler(
            optimizer=nn_optimizer,
            lr1=optimizer_config["nn_initial_lr"],
            lr2=optimizer_config["nn_final_lr"],
            total_steps=num_iters
        )
        
        # segment i
        batch_events = torch.tensor(total_batch_events[i], dtype=torch.float32).to(device)
        batch_events_for_train = torch.tensor(total_batch_events_for_train[i], dtype=torch.float32).to(device)
        gt_flow = torch.tensor(total_batch_gt_flow[i], dtype=torch.float32).to(device)
        batch_t_min, batch_t_max = torch.min(batch_events_for_train[:, 2]), torch.max(batch_events_for_train[:, 2])
        batch_t_ref = torch.linspace(batch_t_min, batch_t_max, 100).to(device)
        indices = torch.arange(100).repeat_interleave(10)
        shuffled_indices = indices[torch.randperm(num_iters)]
        print(f"Segment {i}")

        iter = 0
        early_stopped = False
        actual_iterations = 0
        tools.time_analyzer.start_epoch()
        for j, idx in enumerate(tqdm(shuffled_indices)):
            # reset optimizer
            nn_optimizer.zero_grad()
            spline_optimizer.zero_grad()
            
            # warp events by neural ode
            events_train = batch_events_for_train.clone() # (y, x, t, p)
            events_txy = events_train[..., [2, 1, 0, 3]][:, :3] # (y, x, t, p) ——> (t, x, y, p) ——> (t, x, y)
            t_ref = batch_t_ref[idx]
            # t_ref = warpper.get_reference_time(batch_events_txy, warp_config["tref_setting"])
            warped_events_txy = warpper.warp_events(events_txy, t_ref).unsqueeze(0)

            # create image warped event    
            num_iwe, num_events = warped_events_txy.shape[0], warped_events_txy.shape[1]
            polarity = events_train[:, 3].unsqueeze(0).expand(num_iwe, num_events).unsqueeze(-1)
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
            sl_average_dec_loss = 0
            ssl_average_dec_loss = 0
            
            # CMax loss
            if misc.check_key_and_bool(loss_config, "cmax"):
                if num_iwe == 1:
                    var_loss = criterion["var_criterion"](iwes)
                    grad_loss = criterion["grad_criterion"](iwes)
                elif num_iwe > 1:
                    raise ValueError("Multiple iwes are not supported")
            
            # ssl DEC loss
            if misc.check_key_and_bool(loss_config, "ssl_dec"):
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
                t_min, t_max = torch.min(events_txy[..., 0]), torch.max(events_txy[..., 0])
                interp_t = (t_ref - t_min) / (t_max - t_min)
                lin_vel, ang_vel = motion_spline.forward(interp_t.unsqueeze(0))
                ssl_dec_error, ssl_average_dec_loss = criterion["dec_criterion"](
                    sample_norm_coords, sample_norm_flow, 
                    lin_vel, ang_vel
                )
            
            # sl DEC loss
            if misc.check_key_and_bool(loss_config, "sl_dec"):
                # sample coordinates
                events_mask = tools.imager.create_eventmask(warped_events)
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
                sample_flow = torch.nn.functional.pad(sample_flow, (0, 1), mode='constant', value=0)
                sample_norm_coords = flow_proc.pixel_to_normalized_coords(sample_coords, intrinsic_mat)
                sample_norm_flow = flow_proc.flow_to_normalized_coords(sample_flow, intrinsic_mat)
                
                v_gt, w_gt = gt_lin_vel_spline.evaluate(t_ref), gt_ang_vel_spline.evaluate(t_ref)
                v_gt, w_gt = v_gt.to(torch.float32).unsqueeze(0), w_gt.to(torch.float32).unsqueeze(0)
                sl_dec_error, sl_average_dec_loss = criterion["dec_criterion"](
                    sample_norm_coords, sample_norm_flow, 
                    v_gt, w_gt
                )
                    
            # Total loss
            alpha, beta, gamma = 1, 1, 2.5   # 2.5
            scaled_grad_loss = alpha * grad_loss
            scaled_var_loss = beta * var_loss
            scaled_dec_loss = 0
            if misc.check_key_and_bool(loss_config, "ssl_dec"):
                scaled_dec_loss = gamma * ssl_average_dec_loss
            if misc.check_key_and_bool(loss_config, "sl_dec"):
                scaled_dec_loss = gamma * sl_average_dec_loss
            
            if(j<=200):
                total_loss = - (scaled_grad_loss + scaled_var_loss)
            else:
                total_loss = - (scaled_grad_loss + scaled_var_loss) + scaled_dec_loss
            
            # step
            total_loss.backward()
            nn_optimizer.step()
            nn_scheduler.step()
            if misc.check_key_and_bool(loss_config, "ssl_dec"):
                spline_optimizer.step()
            iter+=1  
            actual_iterations = j + 1
            
            if use_early_stopping:
                current_loss = total_loss.item()
                if early_stopping.step(current_loss):
                    logger.info(f"Early stopping triggered at iteration {j} for segment {i}")
                    logger.info(f"Best training loss: {early_stopping.best:.6f}")
                    logger.info(f"Current loss: {current_loss:.6f}")
                    early_stopped = True
                    break
                
        early_stopping_stats.add_segment_result(i, actual_iterations, early_stopped)
        
        tools.time_analyzer.end_epoch()
        
        flow_metric, motion_metric, eval_motion = run_valid_phase(
            config, i, batch_events, gt_flow, gt_motion, 
            warpper, motion_spline, flow_calculator, tools, device
        )
        
        del warpper.flow_field, motion_spline, nn_optimizer, spline_optimizer, nn_scheduler
        if use_early_stopping:
            del early_stopping
        torch.cuda.empty_cache()
        
        epe.append(flow_metric["EPE"])
        ae.append(flow_metric["AE"])
        out.append(flow_metric["3PE"])
        rmse_lin.append(motion_metric["rmse_linear"])
        rmse_ang.append(motion_metric["rmse_angular"])
        e_lin.append(motion_metric["error_linear"])
        e_ang.append(motion_metric["error_angular"])
        
        all_eval_lin_vel[i] = eval_motion["linear_velocity"]
        all_eval_ang_vel[i] = eval_motion["angular_velocity"]
    
    all_eval_lin_vel_spline = pose.create_vel_cspline(all_eval_lin_vel)
    all_eval_ang_vel_spline = pose.create_vel_cspline(all_eval_ang_vel)
    all_motion_eval_t = torch.from_numpy(gt_lin_vel_array[:, 0]).to(device)
    eval_lin_vel_tensor = all_eval_lin_vel_spline.evaluate(all_motion_eval_t)
    eval_ang_vel_tensor = all_eval_ang_vel_spline.evaluate(all_motion_eval_t)
    
    fig_lin, fig_ang = misc.visualize_velocities(
        config,
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
    
    error_dict = {
        "EPE": np.mean(epe), 
        "AE": np.mean(ae), 
        "3PE": np.mean(out),
        "RMSE_linear": np.mean(rmse_lin),
        "RMSE_angular": np.mean(rmse_ang),
        "ERROR_linear": np.mean(e_lin),
        "ERROR_angular": np.mean(e_ang)
    }
    
    early_stopping_stats.save_to_file(config["logger"]["results_dir"])
    misc.save_metric_as_text(error_dict, config["logger"]["results_dir"])
    stats = tools.time_analyzer.get_statistics()
    return stats

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
    time_stats = run_train_phase(config, dataset, criterions, tools, device)
    
    total_train_time = time_stats["total_train_time"]
    avg_train_time = time_stats["avg_train_time"]
    total_valid_time = time_stats["total_valid_time"]
    avg_valid_time = time_stats["avg_valid_time"]
    # logger.info(f"Total Training Time (min): {total_train_time}")
    # logger.info(f"Average Training Time (min): {avg_train_time}")
    # logger.info(f"Total Validation Time (min): {total_valid_time}")
    # logger.info(f"Average Validation Time (min): {avg_valid_time}")
    
    misc.save_time_log_as_text(time_stats, config["logger"]["results_dir"])
    
    # Save model
    # log_model_path = config["logger"]["model_weight_path"]
    # dir_name = os.path.dirname(log_model_path) 
    # os.makedirs(dir_name, exist_ok=True)
    # torch.save(trained_flow_field.state_dict(), log_model_path)
    
    # tools.wandb_logger.finish()
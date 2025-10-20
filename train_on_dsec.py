import os
import torch 
import torch.optim as optim

from tqdm import tqdm
from typing import Dict
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import DataLoader

from src.utils import (
    misc,
    flow_proc,
    event_proc,
    load_config
)

from src.loader.DSEC.loader import DSECSequence, sequence_collate_fn

from src.loss.focus import FocusLoss
from src.loss.smooth import SparseFlowSmoothnessLoss
from src.loss.dec import DifferentialEpipolarLoss

from src.model.warp import NeuralODEWarpV2
from src.model.eventflow import EventFlowINR, DenseFlowExtractor
from src.model.geometric import Pixel2Cam, CubicBsplineVelocityModel
from src.model.early_stopping import EarlyStopping, EarlyStoppingMode, EarlyStoppingStats
from src.model.scheduler import create_warmup_cosine_scheduler

from src.utils.wandb import WandbLogger
from src.utils.timer import TimeAnalyzer
from src.utils.visualizer import Visualizer
from src.utils.event_imager import EventImager

# logger
from src.utils.logger import get_logger
logger = get_logger()

torch.set_float32_matmul_precision('high')

@dataclass
class Tools:
    viz: Visualizer
    imager: EventImager
    # wandb_logger: WandbLogger
    time_analyzer: TimeAnalyzer
    early_stopping_stats: EarlyStoppingStats

def save_original_iwes(config: Dict, dataset: DSECSequence, loader: DataLoader, tools: Tools, device: torch.device):
    origin_iwe_save_dir = os.path.join(config["logger"]["results_dir"], "origin_iwe")
    for i, sample in enumerate(tqdm(loader, desc=f"Create Original IWE of {dataset.name}...", leave=False)):
        batch_events = sample['events'][0].to(device)
        batch_events_iwe = tools.viz.create_clipped_iwe_for_visualization(batch_events)
        tools.viz.update_save_dir(origin_iwe_save_dir)
        tools.viz.visualize_image(batch_events_iwe.squeeze(0), file_prefix="origin_iwe_")


def run_valid_phase(
    config: Dict, tools: Tools, device: torch.device,
    sample: Dict, 
    batch_events: torch.Tensor,
    warpper: NeuralODEWarpV2,
    flow_calculator: DenseFlowExtractor,
):
    pred_iwe_save_dir = os.path.join(config["logger"]["results_dir"], "pred_iwe")
    pred_flow_save_dir = os.path.join(config["logger"]["results_dir"], "pred_flow")
    submission_flow_save_dir = os.path.join(config["logger"]["results_dir"], "submission_pred_flow")
        
    with torch.no_grad():
        # warp events by trained flow (no clone needed in validation)
        valid_events_txy = batch_events[:, [2, 1, 0]]  # (y,x,t,p) -> (t,x,y), single indexing
        valid_polarity = batch_events[:, 3]  # Extract polarity once

        valid_ref = warpper.get_reference_time(valid_events_txy, "max")
        warped_valid_events_txy = warpper.warp_events(valid_events_txy, valid_ref).unsqueeze(0)

        # create image warped event
        num_iwe_valid, num_events_valid = warped_valid_events_txy.shape[0], warped_valid_events_txy.shape[1]
        valid_polarity_expanded = valid_polarity.unsqueeze(0).expand(num_iwe_valid, num_events_valid).unsqueeze(-1)
        warped_valid_events = torch.cat((warped_valid_events_txy[..., [2,1,0]], valid_polarity_expanded), dim=2)

        # save image warped event
        valid_warped_iwe = tools.viz.create_clipped_iwe_for_visualization(warped_valid_events)
        tools.viz.update_save_dir(pred_iwe_save_dir)
        tools.viz.visualize_image(valid_warped_iwe.squeeze(0), file_prefix="pred_iwe_")

        # predict optical flow (reuse already extracted time values)
        t_start = torch.min(valid_events_txy[:, 0])
        t_end = torch.max(valid_events_txy[:, 0])
        tools.time_analyzer.start_valid()
        pred_flow = flow_calculator.integrate_flow(warpper.flow_field, t_start, t_end).unsqueeze(0) # [B,2,H,W]
        tools.time_analyzer.end_valid()

        # Convert to CPU/numpy once for all subsequent operations
        pred_flow_cpu = pred_flow.squeeze(0).cpu()
        batch_events_cpu = batch_events.cpu().numpy()

        # visualize optical flow
        tools.viz.update_save_dir(pred_flow_save_dir)
        tools.viz.visualize_optical_flow_on_event_mask(pred_flow_cpu.numpy(), batch_events_cpu, file_prefix="pred_flow_masked_")

        # save 16-bit optical flow for dsec eval (reuse pred_flow_cpu to avoid re-conversion)
        tools.viz.update_save_dir(submission_flow_save_dir)
        flow = flow_proc.scale_optical_flow(pred_flow_cpu, 60).numpy()
        file_index = sample['file_index'].item()
        file_name = f'{str(file_index).zfill(6)}.png'
        flow_proc.save_flow(Path(submission_flow_save_dir) / file_name, flow)
    

def run_train_phase(
    config: Dict, tools: Tools, device: torch.device,
    pixel_coords: torch.Tensor,
    batch_idx: int,
    batch_events_for_train: torch.Tensor,
    warpper: NeuralODEWarpV2,
    motion_spline: CubicBsplineVelocityModel,
    criterions: Dict,
):  
    # config for training
    loss_config = config["loss"]
    optimizer_config = config["optimizer"]
    early_stopping_config = config.get("early_stopping", {})
    use_early_stopping = early_stopping_config.get("enabled", False)
    
    # learning rate scheduler
    num_iters = optimizer_config["num_iters"]
    nn_optimizer = optim.AdamW(warpper.flow_field.parameters(), weight_decay=1.0e-5)
    spline_optimizer = optim.AdamW(motion_spline.parameters(), lr=optimizer_config["spline_lr"], weight_decay=1.0e-5)
    nn_scheduler = create_warmup_cosine_scheduler(
        optimizer=nn_optimizer,
        lr_max=optimizer_config["nn_initial_lr"],
        lr_min=optimizer_config["nn_final_lr"],
        total_steps=num_iters,
        warmup_steps=250
    )
    if use_early_stopping:
        early_stopping = EarlyStopping(
            burn_in_steps=early_stopping_config.get("burn_in_steps", 100),
            mode=EarlyStoppingMode.MIN,
            min_delta=early_stopping_config.get("min_delta", 1e-6),
            patience=early_stopping_config.get("patience", 50),
            percentage=False
        )
    
    # sample reference time
    batch_t_min, batch_t_max = torch.min(batch_events_for_train[:, 2]), torch.max(batch_events_for_train[:, 2])
    batch_t_ref = torch.linspace(batch_t_min, batch_t_max, num_iters).to(device)
    indices = torch.arange(num_iters)
    shuffled_indices = indices[torch.randperm(num_iters)]

    # Pre-compute reusable tensors (avoid repeated clone and indexing)
    events_txy = batch_events_for_train[..., [2, 1, 0]]  # (y,x,t,p) -> (t,x,y), no clone needed
    polarity = batch_events_for_train[:, 3]  # Pre-extract polarity once
    t_min_events, t_max_events = torch.min(events_txy[..., 0]), torch.max(events_txy[..., 0])  # Pre-compute time range

    # training loop
    early_stopped = False
    actual_iterations = 0
    tools.time_analyzer.start_epoch()
    for j, idx in enumerate(tqdm(shuffled_indices)):
        # reset optimizer
        nn_optimizer.zero_grad()
        spline_optimizer.zero_grad()

        # warp events by neural ode
        t_ref = batch_t_ref[idx]
        warped_events_txy = warpper.warp_events(events_txy, t_ref).unsqueeze(0)

        # create image warped event
        num_iwe, num_events = warped_events_txy.shape[0], warped_events_txy.shape[1]
        polarity_expanded = polarity.unsqueeze(0).expand(num_iwe, num_events).unsqueeze(-1)
        warped_events = torch.cat((warped_events_txy[..., [2,1,0]], polarity_expanded), dim=2) # (t,x,y) -> (y,x,t,p)
        
        iwes = tools.imager.create_iwes(
            events=warped_events,
            method="bilinear_vote",
            sigma=0,
            blur=True
        ) # [n,h,w] n should be one
        
        grad_loss = 0
        var_loss = 0
        ssl_average_dec_loss = 0
        smooth_loss = 0

        # CMax loss
        if misc.check_key_and_bool(loss_config, "cmax"):
            if num_iwe == 1:
                var_loss = criterions["var_criterion"](iwes)
                grad_loss = criterions["grad_criterion"](iwes)
            elif num_iwe > 1:
                raise ValueError("Multiple iwes are not supported")

        # Shared coordinate sampling for ssl_dec and smooth losses
        need_sampling = misc.check_key_and_bool(loss_config, "ssl_dec") or misc.check_key_and_bool(loss_config, "smooth")
        if need_sampling:
            # Sample coordinates once and reuse for both losses
            events_mask = tools.imager.create_eventmask(warped_events)
            events_mask = events_mask[0].squeeze(0)
            sample_coords = pixel2cam_converter.sample_sparse_coordinates(
                coord_tensor=pixel_coords, mask=events_mask, n=10000
            )
            t_ref_expanded = t_ref * torch.ones(sample_coords.shape[1], device=device).reshape(1,-1,1)
            sample_txy = torch.cat((t_ref_expanded, sample_coords[...,0:2]), dim=2)
            sample_flow = warpper.flow_field.forward(sample_txy)

        # ssl DEC loss
        if misc.check_key_and_bool(loss_config, "ssl_dec"):
            # Reuse already computed sample_coords, sample_txy, and sample_flow
            sample_flow_padded = torch.nn.functional.pad(sample_flow, (0, 1), mode='constant', value=0)
            sample_norm_coords = flow_proc.pixel_to_normalized_coords(sample_coords, intrinsic_mat)
            sample_norm_flow = flow_proc.flow_to_normalized_coords(sample_flow_padded, intrinsic_mat)

            # differential epipolar constrain (use pre-computed time range)
            interp_t = (t_ref - t_min_events) / (t_max_events - t_min_events)
            lin_vel, ang_vel = motion_spline.forward(interp_t.unsqueeze(0))
            ssl_dec_error, ssl_average_dec_loss = criterions["dec_criterion"](
                sample_norm_coords, sample_norm_flow, lin_vel, ang_vel
            )

        # Flow smoothness loss
        if misc.check_key_and_bool(loss_config, "smooth"):
            # Reuse sample_txy and sample_flow already computed above
            smooth_loss = criterions["smooth_criterion"](sample_txy, sample_flow[:, :, :2], warpper.flow_field)
                
        # Total loss
        alpha, beta, gamma, delta = 1, 1, 2.5, 0.2
        scaled_grad_loss = alpha * grad_loss
        scaled_var_loss = beta * var_loss
        scaled_dec_loss = 0
        scaled_smooth_loss = 0

        if misc.check_key_and_bool(loss_config, "ssl_dec"):
            scaled_dec_loss = gamma * ssl_average_dec_loss

        if misc.check_key_and_bool(loss_config, "smooth"):
            scaled_smooth_loss = delta * smooth_loss

        if(j<=200):
            total_loss = - (scaled_grad_loss + scaled_var_loss)
        else:
            total_loss = - (scaled_grad_loss + scaled_var_loss) + scaled_dec_loss + scaled_smooth_loss
        
        # step
        total_loss.backward()
        nn_optimizer.step()
        nn_scheduler.step()
        if misc.check_key_and_bool(loss_config, "ssl_dec"):
            spline_optimizer.step()
        actual_iterations = j + 1 
        
        if use_early_stopping:
            current_loss = total_loss.item()
            if early_stopping.step(current_loss):
                early_stopped = True
                break
            
    tools.early_stopping_stats.add_batch_result(batch_idx, actual_iterations, early_stopped)
    tools.time_analyzer.end_epoch()
    
    del nn_optimizer, spline_optimizer, nn_scheduler
    if use_early_stopping:
        del early_stopping
    torch.cuda.empty_cache()
    
    return warpper, motion_spline
    

if __name__ == '__main__':
    # load configs
    args = load_config.parse_args()
    config = load_config.load_yaml_config(args.config)
    data_config = config["data"]
    model_config = config["model"]
    warp_config = config["warp"]
    
    # detect cuda
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    logger.info(f"Use device: {device}")
    
    # fix seed
    misc.fix_random_seed()

    # load dataset 
    seq_path = Path(data_config["data_path"]) / data_config["sequence"]
    timestamp_path = Path(data_config["timestamp_path"]) / f'{data_config["sequence"]}.csv'
    dataset = DSECSequence(seq_path=seq_path, phase='test', timestamp_path=timestamp_path)
    image_size = (data_config["hight"], data_config["width"])
    
    # generate coordinates on image plane
    intrinsic_mat = torch.from_numpy(dataset.intrinsic).clone()
    pixel2cam_converter = Pixel2Cam(dataset.height, dataset.width, intrinsic_mat, device)
    image_coords = pixel2cam_converter.generate_image_coordinate()
    flow_calculator = DenseFlowExtractor(grid_size=image_size, device=device, warp_config=warp_config)

    # instantiate criterion
    dec_criterion = DifferentialEpipolarLoss()
    var_criterion = FocusLoss(loss_type="variance", norm="l1")
    grad_criterion = FocusLoss(loss_type="gradient_magnitude", norm="l1")
    smooth_criterion = SparseFlowSmoothnessLoss()
    criterions = {
        "grad_criterion": grad_criterion,
        "var_criterion": var_criterion,
        "dec_criterion": dec_criterion,
        "smooth_criterion": smooth_criterion
    } 
    
    # event2img converter
    imager = EventImager(image_size)

    # Visualizer
    viz = Visualizer(
        image_shape=image_size, show=False, save=True, save_dir=config["logger"]["results_dir"],
    )

    # time analysis
    time_analyzer = TimeAnalyzer()

    # Stats of early stopping
    early_stopping_stats = EarlyStoppingStats(
        batch_iterations=[],
        early_stopped_batchs=[],
        total_batchs=0
    )
    
    # Wandb
    # wandb_logger = WandbLogger(config)
    
    # create tools
    tools = Tools(
        viz=viz,
        imager=imager,
        # wandb_logger=wandb_logger,
        time_analyzer=time_analyzer,
        early_stopping_stats=early_stopping_stats
    )
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    tools.early_stopping_stats.total_batchs = len(loader)
    
    # save original iwes
    save_original_iwes(config, dataset, loader, tools, device)
    
    for i, sample in enumerate(tqdm(loader, desc=f"Training on {dataset.name}...", leave=False)):
        # prepare data
        batch_events = sample['events'][0].to(device)
        batch_t_min, batch_t_max = torch.min(batch_events[:, 2]), torch.max(batch_events[:, 2])

        # reset model
        flow_field = EventFlowINR(model_config).to(device)
        warpper = NeuralODEWarpV2(flow_field, device, warp_config)
        motion_spline = CubicBsplineVelocityModel().to(device)
        
        # trainer
        # logger.info(f"Training on batch {i}...")
        warpper, motion_spline = run_train_phase(
            config, tools, device,
            pixel_coords=image_coords,
            batch_idx=i,
            batch_events_for_train=batch_events,
            warpper=warpper,
            motion_spline=motion_spline,
            criterions=criterions
        )
        
        # evaluation
        # logger.info(f"Evaluation on batch {i}...")
        run_valid_phase(
            config, tools, device,  
            sample=sample,
            batch_events=batch_events, 
            warpper=warpper, 
            flow_calculator=flow_calculator
        )
        
        # delete last model to release GPU memory
        del warpper, motion_spline
        torch.cuda.empty_cache()
        
    tools.early_stopping_stats.save_to_file(config["logger"]["results_dir"])
    time_stats = tools.time_analyzer.get_statistics()  
    misc.save_time_log_as_text(time_stats, config["logger"]["results_dir"])
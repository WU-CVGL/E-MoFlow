import os
import sys
import torch 
import logging
import cv2 as cv2
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from typing import Dict

from src.loss.focus import FocusLoss
from src.loss.motion import MotionLoss
from src.utils import misc
from src.utils import pose
from src.utils import event_proc
from src.utils import metric
from src.utils import vector_math
from src.utils import load_config
from src.model.eventflow import EventFlowINR
from src.utils.wandb import WandbLogger
from src.model.warp import NeuralODEWarp
from src.model import geometric
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
    # Split events stream to extract data for validation
    eval_frame_timestamp_list = dataset.eval_frame_time_list()
    eval_dt_list = [4]
    total_batch_events = []
    total_gt_flow = []
    for eval_dt in tqdm(eval_dt_list, desc=f"Split the events into batches for validation", leave=True):
        for i1 in tqdm(
            range(len(eval_frame_timestamp_list) - eval_dt), 
            desc=f"Divide the event stream into {eval_dt}-frame intervals", leave=True
        ):  
            t1 = eval_frame_timestamp_list[i1]
            t2 = eval_frame_timestamp_list[i1 + eval_dt]
            
            gt_flow = dataset.load_optical_flow(t1, t2)
            ind1 = dataset.time_to_index(t1)  # event index
            ind2 = dataset.time_to_index(t2)
            batch_events = dataset.load_event(ind1, ind2)
            batch_events[...,2] -= dataset.min_ts # norm_t
            # batch_events[...,2] = (batch_events[...,2] - min(batch_events[...,2])) / (max(batch_events[...,2]) - min(batch_events[...,2]))
            # batch_events[...,2] = (batch_events[...,2] - dataset.min_ts) / dataset.data_duration
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
        # flow_field.to(torch.float64)
        warpper = NeuralODEWarp(flow_field, device, **warp_config)
        optimizer = optim.Adam(flow_field.parameters(), lr=optimizer_config["initial_lrate"])
        
        K_tensor = torch.Tensor(
            [[226.38018519795807, 0, 173.6470807871759],
             [0, 226.15002947047415, 133.73271487507847],
             [0, 0, 1]]).to(device)
        
        # decay lr
        decay_steps = 1
        num_epochs = config["optimizer"]["num_epoch"]
        final_lrate = config["optimizer"]["final_lrate"]
        initial_lrate = config["optimizer"]["initial_lrate"]
        decay_factor = (final_lrate / initial_lrate) ** (1 / (num_epochs / decay_steps))
        print(f"learning rate decay factor:{decay_factor}")
        initial_lr = optimizer.param_groups[0]['lr']
        print(f"Segment {i}, Initial LR: {initial_lr}")
        iter = 0

        # if(i!=1825):
        #     continue
        for _ in tqdm(range(num_epochs)):
            time_analyzer.start_epoch()
            
            optimizer.zero_grad()
            
            batch_events = valid_events_origin.detach().clone() # (y, x, t, p)
            origin_iwe = imager.create_iwes(batch_events)
            batch_events = batch_events[..., [2, 1, 0, 3]]  # (t, x, y, p)
            batch_events_txy = batch_events[:, :3]
            
            # get t_ref
            t_ref = warpper.get_reference_time(batch_events_txy, warp_config["tref_setting"])

            # warp by neural ode 
            warped_events_txy = warpper.warp_events(batch_events_txy, t_ref, method=warp_config["solver"])
            # warped_events_txy = warpper.warp_events(batch_events_txy, t_ref).unsqueeze(0)

            # create image warped event    
            num_iwe, num_events = warped_events_txy.shape[0], warped_events_txy.shape[1]
            polarity = batch_events[:, 3].unsqueeze(0).expand(num_iwe, num_events).unsqueeze(-1)
            warped_events = torch.cat((warped_events_txy[..., [1,2,0]], polarity), dim=2).to(device)
            warped_events = warped_events[...,[1,0,2,3]] # (x,y,t,p) ——> (y,x,t,p)
            
            # events should be reorganized as (y,x,t,p) that can processed by create_iwes
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
                
            # Total loss
            alpha, beta, gamma = 1, 1, 10
            scaled_grad_loss = alpha * grad_loss
            scaled_var_loss = beta * var_loss
            total_loss = - (scaled_grad_loss + scaled_var_loss)
            
            # step
            total_loss.backward()
            optimizer.step()
            
            # update lr
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_factor
            iter+=1  

        # valid
        time_analyzer.start_valid()
        with torch.no_grad():
            valid_events = valid_events_origin.detach().clone()
            valid_events = valid_events[:, [2, 1, 0, 3]] # (y, x, t, p) ——> (t, x, y, p)
            valid_events_txy = valid_events[:, :3]
            
            valid_ref = warpper.get_reference_time(valid_events_txy, "min")
            warped_valid_events_txy = warpper.warp_events(valid_events_txy, valid_ref, method=warp_config["solver"])
            # warped_valid_events_txy = warpper.warp_events(valid_events_txy, valid_ref).unsqueeze(0)
            
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
                intrinsic_mat = K_tensor,
                model=warpper.flow_field, 
                normalize_coords_mode="None", 
                device=device
            )
            eval_t = eval_frame_timestamp_list[i] - dataset.min_gray_ts
            eval_dt = eval_frame_timestamp_list[i + 4] - eval_frame_timestamp_list[i]
            U, V, _, _ = flow_calculator.extract_flow_from_inr(eval_t + 0.5 * eval_dt, 1)
            flow = torch.stack((V, U), dim=2).permute(2, 0, 1)
            flow = flow.cpu().numpy() * eval_dt
            # flow = flow.cpu().numpy() 
            viz.visualize_optical_flow_on_event_mask(flow, valid_events_origin.cpu().numpy(), file_prefix="pred_flow_masked" + str(i))
            
            pred_flow = torch.from_numpy(flow).unsqueeze(0).to(device)
            gt_flow = torch.from_numpy(gt_flow).unsqueeze(0).to(device)
            event_mask = imager.create_eventmask(valid_events_origin).to(device)
            flow_error, _ = metric.calculate_flow_error(gt_flow, pred_flow, event_mask=event_mask)  # type: ignore
            
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
    
    logger.info(f"Average EPE: {np.mean(epe)}")
    logger.info(f"Average AE: {np.mean(ae)}")
    logger.info(f"Average 3PE: {np.mean(out)}")
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
    
    # create model
    # flow_field: EventFlowINR = EventFlowINR(model_config).to(device)
    # warpper = NeuralODEWarp(flow_field, device, **warp_config)
    
    # # create NN optimizer
    # optimizer = optim.Adam(flow_field.parameters(), lr=optimizer_config["initial_lrate"])
    
    # instantiate criterion
    grad_criterion = FocusLoss(loss_type="gradient_magnitude", norm="l1")
    var_criterion = FocusLoss(loss_type="variance", norm="l1")
    motion_criterion = MotionLoss(loss_type="MSE")
    criterions = {
        "grad_criterion": grad_criterion, 
        "var_criterion": var_criterion,
        "motion_criterion": motion_criterion
    } 
    
    # display origin valid data
    # valid_origin_iwe = viz.create_clipped_iwe_for_visualization(valid_events)
    # wandb_logger.write_img("valid_iwe", valid_origin_iwe)
    # wandb_logger.update_buffer()

    # time analysis
    time_analyzer = TimeAnalyzer()

    # train
    # trained_flow_field, time_stats = run_train_phase(
    #     args, config, dataset, warpper, optimizer, criterions,
    #     imager, viz, wandb_logger, time_analyzer, device
    # )
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
    
    
    


    
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
    config: Dict, 
    data: torch.Tensor, 
    warpper: NeuralODEWarp,
    nn_optimizer: torch.optim,
    criterion: Dict,
    imager: EventImageConverter,
    viz: Visualizer,
    wandb_logger: WandbLogger,
    time_analyzer: TimeAnalyzer,
    device: torch.device
):  
    train_events, split_timestamps = data["train_events"], data["split_timestamps"]
    train_events = torch.tensor(train_events[1402], dtype=torch.float32).to(device)
    train_events = train_events.unsqueeze(0)
    valid_events_origin = torch.tensor(train_events[0], dtype=torch.float32).to(device)
    print(valid_events_origin.shape)
    valid_origin_iwe = viz.create_clipped_iwe_for_visualization(valid_events_origin)
    wandb_logger.write_img("valid_iwe", valid_origin_iwe)
    wandb_logger.update_buffer()
    K = torch.Tensor([[226.38018519795807, 0, 173.6470807871759],
                      [0, 226.15002947047415, 133.73271487507847],
                      [0, 0, 1]]).to(device)
    # decay lr
    decay_steps = 1
    num_epochs = config["optimizer"]["num_epoch"]
    final_lrate = config["optimizer"]["final_lrate"]
    initial_lrate = config["optimizer"]["initial_lrate"]
    decay_factor = (final_lrate / initial_lrate) ** (1 / (num_epochs / decay_steps))
    iter = 0
    
    # epoch
    for i in tqdm(range(num_epochs)):
        time_analyzer.start_epoch()
        
        # batch
        for batch_idx in tqdm(range(len(train_events)), desc=f"Tranning {i} epoch", leave=True):
            nn_optimizer.zero_grad()
            
            # select batch events in window  (y, x, t, p)
            batch_events = torch.tensor(train_events[batch_idx], dtype=torch.float32).to(device)
            batch_events = batch_events[..., [2, 1, 0, 3]]  # (t, x, y, p)
            batch_events_txy = batch_events[:, :3]
            
            # get t_ref
            t_ref = warpper.get_reference_time(batch_events_txy, warp_config["tref_setting"])

            # warp by neural ode 
            warped_events_txy = warpper.warp_events(batch_events_txy, t_ref, method=warp_config["solver"])
            # warped_norm_txy = warpper.warp_events(norm_txy, t_ref).unsqueeze(0)

            # create image warped event    
            num_iwe, num_events = warped_events_txy.shape[0], warped_events_txy.shape[1]
            polarity = batch_events[:, 3].unsqueeze(0).expand(num_iwe, num_events).unsqueeze(-1)
            warped_events = torch.cat((warped_events_txy[..., [1,2,0]], polarity), dim=2)
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
            
            # Total loss
            alpha, beta, gamma = 1, 1, 10
            scaled_grad_loss =  alpha * grad_loss
            scaled_var_loss = beta * var_loss
            total_loss = 1 / (scaled_grad_loss + scaled_var_loss)
            
            # step
            total_loss.backward()
            nn_optimizer.step()
        
        # update lr
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_factor
            
        tqdm.write(
            f"[LOG] Epoch {iter} Total Loss: {total_loss.item()}, Var Loss: {scaled_var_loss.item()}, Grad Loss: {scaled_grad_loss.item()}"
        )    
        wandb_logger.write("var_loss", scaled_var_loss.item())
        wandb_logger.write("grad_loss", scaled_grad_loss.item())
        wandb_logger.write("total_loss", total_loss.item())
        wandb_logger.write("learing_rate", param_group['lr'])
        
        # valid
        if (iter+1) % 10 == 0:
            time_analyzer.start_valid()
            with torch.no_grad():
                valid_events = valid_events_origin[:, [2, 1, 0, 3]] # (y, x, t, p) ——> (t, x, y, p)
                valid_events_txy = valid_events[:, :3]
                
                valid_ref = warpper.get_reference_time(valid_events_txy, "min")
                warped_valid_events_txy = warpper.warp_events(valid_events_txy, valid_ref, method=warp_config["solver"])
                # valid_warped_norm_txy = warpper.warp_events(valid_norm_txy, valid_ref).unsqueeze(0)
                
                # create image warped event 
                num_iwe_valid, num_events_valid = warped_valid_events_txy.shape[0], warped_valid_events_txy.shape[1]       
                valid_polarity = valid_events[:, 3].unsqueeze(0).expand(num_iwe_valid, num_events_valid).unsqueeze(-1).to(device)
                warped_valid_events = torch.cat((warped_valid_events_txy[..., [1,2,0]], valid_polarity), dim=2)
                warped_valid_events = warped_valid_events[...,[1,0,2,3]]
                
                # log
                valid_warped_iwe = viz.create_clipped_iwe_for_visualization(warped_valid_events)
                wandb_logger.write_img("valid_iwe", valid_warped_iwe)
            time_analyzer.end_valid()
        
        # log
        time_analyzer.end_epoch()
        wandb_logger.update_buffer()
        iter+=1
    
    logger.info("Training phase completed!")
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
    events = dataset.get_events_for_train()           # Note: x is hight, y is width
    # valid_events = dataset.get_events_for_valid(t1=10, t2=10.1)
    eval_frame_timestamp_list = dataset.eval_frame_time_list()
    eval_dt_list = [4]
    total_batch_events = []
    for eval_dt in tqdm(eval_dt_list, desc=f"Split the events into batches for training", leave=True):
        for i1 in tqdm(
            range(len(eval_frame_timestamp_list) - eval_dt), 
            desc=f"Divide the event stream into {eval_dt}-frame intervals", leave=True
        ):  
            t1 = eval_frame_timestamp_list[i1]
            t2 = eval_frame_timestamp_list[i1 + eval_dt]

            ind1 = dataset.time_to_index(t1)  # event index
            ind2 = dataset.time_to_index(t2)
            batch_events = dataset.load_event(ind1, ind2)
            batch_events[...,2] -= dataset.min_ts # norm_t
            total_batch_events.append(batch_events)
    # valid_events = total_batch_events[7000]
    # valid_events = torch.from_numpy(valid_events).float().to(device)
    data = {"train_events": total_batch_events, "split_timestamps": eval_frame_timestamp_list}
    logger.info(f"{len(total_batch_events)} batches of training data are ready.")
    # valid_events = dataset.get_events_for_valid(t1=62.6808,t2=63.96)
    # events[...,2] = (events[...,2] - dataset.min_ts) / dataset.data_duration # norm t
    # valid_events[...,2] = (valid_events[...,2] - dataset.min_ts) / dataset.data_duration
    # events[...,2] = events[...,2] - dataset.min_ts
    # valid_events[...,2] = valid_events[...,2] - dataset.min_ts
    # events = torch.from_numpy(events).float().to(device)
    # valid_events = torch.from_numpy(valid_events).float().to(device)
    
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
    flow_field: EventFlowINR = EventFlowINR(model_config).to(device)
    warpper = NeuralODEWarp(flow_field, device, **warp_config)
    
    # create NN optimizer
    optimizer = optim.Adam(flow_field.parameters(), lr=optimizer_config["initial_lrate"])
    
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
    trained_flow_field, time_stats = run_train_phase(
        config, data, warpper, optimizer, criterions,
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
    
    
    


    
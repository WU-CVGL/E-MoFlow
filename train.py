import os
import sys
import time
import random
import logging
import cv2 as cv2
import numpy as np

import torch 
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from src.loss import focus
from src.utils import load_config
from src.model.inr import EventFlowINR
from src.utils.wandb import WandbLogger
from src.model.warp import NeuralODEWarp
from src.utils.timer import TimeAnalyzer
from src.event_data import EventStreamData
from src.dataset_loader import dataset_manager
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

# set seeds
def fix_random_seed(seed_idx=666) -> None:
    random.seed(seed_idx)
    np.random.seed(seed_idx)
    torch.manual_seed(seed_idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_idx)
        torch.cuda.manual_seed_all(seed_idx)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    fix_random_seed()

    # load dataset 
    dataset_name = data_config.pop('dataset_name')
    dataloader_config = data_config.pop('loader')
    dataset = dataset_manager.get_dataset(dataset_name, data_config)
    loader, provider = dataset_manager.create_loader(dataset, dataloader_config)
    
    # event2img converter
    image_size = (data_config["hight"], data_config["weight"])
    converter = EventImageConverter(image_size)

    # create model
    flow_field = EventFlowINR(model_config).to(device)
    warpper = NeuralODEWarp(flow_field, device, **warp_config)

    # create optimizer
    optimizer = optim.Adam(flow_field.parameters(), lr=optimizer_config["initial_lrate"])

    # prepare data to valid
    valid_data = provider.get_valid_data(config["valid"]["file_idx"])
    valid_events = valid_data["events"]
    valid_events_norm = valid_data["events_norm"]
    valid_events_timestamps = valid_data["timestamps"]
    valid_batch_txy = valid_events_norm[:, :-1].float().to(device)

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
            # get batch data
            events = sample["events"].squeeze(0)
            events_norm = sample["events_norm"].squeeze(0)
            timestamps = sample["timestamps"].squeeze(0)
            batch_txy = events_norm[:, :-1].float().to(device)
            origin_iwe = converter.create_iwes(events[:, [2,1,0,3]])

            # get t_ref
            ref = warpper.get_reference_time(batch_txy, warp_config["tref_setting"])

            # odewarp 
            warped_batch_txy = warpper.warp_events(batch_txy, ref, method=warp_config["solver"])

            # create image warped event    
            num_iwe = warped_batch_txy.shape[0]
            num_events = warped_batch_txy.shape[1]
            polarity = events[:, 3].unsqueeze(0).expand(num_iwe, num_events).unsqueeze(-1).to(device)
            warped_events_xytp = torch.cat((warped_batch_txy[..., [2,1,0]], polarity), dim=2)
            warped_events_xytp[..., :2] *= torch.Tensor(image_size).to(device)
            iwes = converter.create_iwes(
                events=warped_events_xytp,
                method="bilinear_vote",
                sigma=1.0,
                blur=True
            ) # [n,h,w] n can be one or multi

            # loss
            if num_iwe == 1:
                # var_loss = torch.Tensor([0]).to(device)
                var_loss = focus.calculate_focus_loss(iwes, loss_type='variance')
                grad_loss = focus.calculate_focus_loss(iwes, loss_type='gradient_magnitude', norm='l1')
            elif num_iwe > 1:
                var_loss = torch.Tensor([0]).to(device)
                # var_loss = focus.calculate_focus_loss(iwe, loss_type='variance')
                grad_loss_t_min = focus.calculate_focus_loss(iwes[0].unsqueeze(0), loss_type='gradient_magnitude', norm='l1')
                grad_loss_t_mid = focus.calculate_focus_loss(iwes[len(iwes)//2].unsqueeze(0), loss_type='gradient_magnitude', norm='l1')
                grad_loss_t_max = focus.calculate_focus_loss(iwes[-1].unsqueeze(0), loss_type='gradient_magnitude', norm='l1')
                gard_loss_origin = focus.calculate_focus_loss(origin_iwe, loss_type='gradient_magnitude', norm='l1')

                grad_loss = (grad_loss_t_min + grad_loss_t_max + 2 * grad_loss_t_mid) / 4 * gard_loss_origin
            total_loss = - (grad_loss + var_loss)
            wandb_logger.write("var_loss", var_loss.item())
            wandb_logger.write("grad_loss", grad_loss.item())
            wandb_logger.write("train_loss", total_loss.item())

            # optimize
            optimizer.zero_grad()
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
                valid_ref = warpper.get_reference_time(valid_batch_txy, "min")
                valid_warped_batch_txy = warpper.warp_events(valid_batch_txy, valid_ref, method=warp_config["solver"])
                # create image warped event 
                num_iwe_valid = valid_warped_batch_txy.shape[0]
                num_events_valid = valid_warped_batch_txy.shape[1]          
                valid_polarity = valid_events[:, 3].unsqueeze(0).expand(num_iwe_valid, num_events_valid).unsqueeze(-1).to(device)
                valid_warped_batch_txy = torch.cat((valid_warped_batch_txy[..., [2,1,0]], valid_polarity), dim=2)
                valid_warped_batch_txy[..., :2] *= torch.Tensor(image_size).to(device)
                optimized_iwe = converter.create_iwes(
                    events=valid_warped_batch_txy,
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

    # save model
    log_model_path = config["logger"]["model_weight_path"]
    dir_name = os.path.dirname(log_model_path) 
    os.makedirs(dir_name, exist_ok=True)
    torch.save(flow_field.state_dict(), log_model_path)



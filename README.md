<h1 align="center"> 🥺 E-MoFlow: Learning Egomotion and Optical Flow from Event Data via Implicit Regularization  </h1>

<div style="height:-20px;"></div>

<div align="center"><h3>[NeurIPS 2025]</h3></div>
<div align="center">
    <a href="https://akawincent.github.io/">Wenpu Li*</a>
    &nbsp;·&nbsp;
    <a href="https://bangyan101.github.io/">Bangyan Liao*</a>
    &nbsp;·&nbsp;
    <a href="https://sites.google.com/view/zhouyi-joey/home">Yi Zhou</a>
    &nbsp;·&nbsp;
    <a href="https://github.com/insomniaaac">Qi Xu</a>
    &nbsp;·&nbsp;
    <a href="https://i.rm2.net/">Pian Wan</a>
    &nbsp;·&nbsp;
    <a href="https://ethliup.github.io/">Peidong Liu†</a>

<!-- <a href="https://arxiv.org/abs/2505.21060">Paper</a>|  -->

<h3 align="center"><a href="https://akawincent.github.io/EMoFlow/">Project Page</a> </h3>


<div style="width:80%;margin:auto;">
    <img src="assets/emoflow_teaser_bg.jpg" style="width:80%;" />
    <div style="height:40px;"></div>
    <img src="assets/indoor_flying_1_dt_4_web.gif" style="width:80%;" />
    <img src="assets/outdoor_day_1_dt_4_web.gif" style="width:80%;" />
</div>

</div>

</br>

> This repository is an official PyTorch implementation of the paper "E-MoFlow: Learning Egomotion and Optical Flow from Event Data via Implicit Regularization". We reavel that implicit regularizations can enable the mutual promotion of self-supervised learning for optical flow and egomotion.  Importantly, our method is capable of simultaneously learning the optical flow field and camera motion solely from event data.


## 📢 News & 🚧 TODO
- [ ] Add point tracking visualization results.
- [ ] Use uv to manage project environments and dependencies.
- [ ] Publish the v1 paper on arXiv.
- [ ] Release the training code for DSEC dataset、
- `2025.10.13` Training code for MVSEC dataset has been released. 
- `2025.10.12` Our project homepage is available online.
- `2025.09.18` Our paper was accepted by NeurIPS2025!! Congratulates to all collaborators!!

## ⚙️ Installation
Working in progress...

## 📊 Data Preparation
We conducted experiments on the [MVSEC](https://daniilidis-group.github.io/mvsec/) dataset and [DSEC](https://dsec.ifi.uzh.ch/) dataset. After downloading the datasets, for MVSEC, you only need to modify the `data_path` and `gt_path` in the config file to your paths. For DSEC, you only need to modify the `data_path` and `timestamp_path` in the config file to your paths.

The download link is as follows:
- MVSEC: [[hdf5](https://drive.google.com/drive/folders/1rwyRk26wtWeRgrAx_fgPc-ubUzTFThkV)] [[gt flow](https://drive.google.com/drive/folders/1XS0AQTuCwUaWOmtjyJWRHkbXjj_igJLp)]
- DSEC: [[hdf5](https://download.ifi.uzh.ch/rpg/DSEC/test_coarse/test_events.zip)] [[timestamps](https://download.ifi.uzh.ch/rpg/DSEC/test_forward_optical_flow_timestamps.zip)]

## 🚀 Run
You can run E-MoFlow on the MVSEC dataset and DSEC dataset by:
```bash
python train_on_mvsec.py --gpu <gpu_idx> --config <config_file_path>
python train_on_dsec.py --gpu <gpu_idx> --config <config_file_path>
```
Additionally, you can modify the config file to conduct ablation studies or enable early stopping strategies to achieve a trade-off between speed and accuracy.

After running, the following results will be output:
```
outputs/
├── project/
│   ├── expname/
|   |   |-- origin_iwe                  # Original IWE
|   |   |-- pred_iwe                    # Warped IWE using predicted optical flow
|   |   |-- pred_flow                   # Predicted optical flow
|   |   |-- submission_pred_flow        # Predicted optical flow for DSEC evaluation (only for DSEC)
|   |   |-- gt_flow                     # Ground truth optical flow (only for MVSEC)
|   |   |-- motion                      # Estimated egomotion (only for MVSEC)
|   |   |-- metric.txt                  # Quantitative results (only for MVSEC)
|   |   |-- time_stats.txt              # Runing time statistics
|   |   |-- early_stopping_stats.txt    # Early stopping statistics (only for early stopping experiments)
```

## 📖 Citation

If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{li2025emoflow,
      author = {Wenpu Li and Bangyan Liao and Yi Zhou and Qi Xu and Pian Wan and Peidong Liu},
      title = {E-MoFlow: Learning Egomotion and Optical Flow from Event Data via Implicit Regularization},
      booktitle = {Annual Conference on Neural Information Processing Systems (NeurIPS)},
      year = {2025}
  } 

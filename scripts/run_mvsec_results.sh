#!/bin/bash

commands=(
    "python train_on_mvsec.py --gpu 0 --config './configs/mvsec/dt_1/indoor_flying_1_dt_1.yaml' &"
    "python train_on_mvsec.py --gpu 1 --config './configs/mvsec/dt_1/indoor_flying_2_dt_1.yaml' &"
    "python train_on_mvsec.py --gpu 2 --config './configs/mvsec/dt_1/indoor_flying_3_dt_1.yaml' &"
    "python train_on_mvsec.py --gpu 3 --config './configs/mvsec/dt_1/outdoor_day_1_dt_1.yaml' &"
    "python train_on_mvsec.py --gpu 4 --config './configs/mvsec/dt_4/indoor_flying_1_dt_4.yaml' &"
    "python train_on_mvsec.py --gpu 5 --config './configs/mvsec/dt_4/indoor_flying_2_dt_4.yaml' &"
    "python train_on_mvsec.py --gpu 6 --config './configs/mvsec/dt_4/indoor_flying_3_dt_4.yaml' &"
    "python train_on_mvsec.py --gpu 7 --config './configs/mvsec/dt_4/outdoor_day_1_dt_4.yaml' &"
)

for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait


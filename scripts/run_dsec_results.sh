#!/bin/bash

commands=(
    "python train_on_dsec.py --gpu 1 --config './configs/dsec/full_model/interlaken_00_b.yaml' &"
    "python train_on_dsec.py --gpu 2 --config './configs/dsec/full_model/interlaken_01_a.yaml' &"
    "python train_on_dsec.py --gpu 3 --config './configs/dsec/full_model/thun_01_a.yaml' &"
    "python train_on_dsec.py --gpu 4 --config './configs/dsec/full_model/thun_01_b.yaml' &"
    "python train_on_dsec.py --gpu 5 --config './configs/dsec/full_model/zurich_city_12_a.yaml' &"
    "python train_on_dsec.py --gpu 6 --config './configs/dsec/full_model/zurich_city_14_c.yaml' &"
    "python train_on_dsec.py --gpu 7 --config './configs/dsec/full_model/zurich_city_15_a.yaml' &"
)

for cmd in "${commands[@]}"; do
    eval "$cmd"
done

wait


#!/bin/bash

CONFIG_DIR="configs/Germany/anytime" # model parameter config folder | 配置文件所在文件夹
WORK_DIR="work_dir/Germany/anytime" # model and log saving dir | 模型运行log和推理结果保存文件夹
ORIGINAL_DATASET_PATH="dataset/dataset_for_model/Germany/hdf/anytime.hdf" # path of original hdf dataset | HDF数据集文件路径

# Used to obtain geographic coordinate information when writing out imputed images | 原始TIF文件路径, 一景即可, 用于在写出重建结果TIF时获取参考投影信息等
REF_TIF_PATH="dataset/dataset_for_model/Germany/tif/40%/S2_L2A_20190208.tif" 

# model parameter config name| 配置文件名
CONFIG_FILES=(
    "AnytimeFormer-Germany-40%-r8-128-wTV.yaml" # with total variation loss
)

for config_file in "${CONFIG_FILES[@]}"; do
    python main.py \
        --config_path "$CONFIG_DIR/$config_file" \
        --work_dir $WORK_DIR \
        --original_dataset_path $ORIGINAL_DATASET_PATH \
        --ref_tif_path $REF_TIF_PATH \
        --mode "train_test_anytime" \
        --debug_mode False
done

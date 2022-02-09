#!/usr/bin/env bash

EXPNAME=$1
SAVEDIR=checkpoints/coco/${EXPNAME}

# ------------------------------- Base Pre-train ---------------------------------- #
# Normal Pre-train (2x)
#python3 -m tools.train_net --num-gpus 8 \
#        --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_base_2x.yaml \
#        --opts OUTPUT_DIR ${SAVEDIR}/frcn_r101_base

# for MFA Pre-train
python3 -m tools.train_net --num-gpus 8 \
        --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_base_mfa.yaml \
        --opts OUTPUT_DIR ${SAVEDIR}/frcn_r101_base


# -------------------------------  FSOD (only novel 20 class) ---------------------------------- #
# ------------------------------ Model Preparation -------------------------------- #
python3 -m tools.ckpt_surgery \
        --src ${SAVEDIR}/frcn_r101_base/model_final.pth \
        --method remove \
        --save-dir ${SAVEDIR}/frcn_r101_base \
        --dataset coco
BASE_WEIGHT=${SAVEDIR}/frcn_r101_base/model_reset_remove.pth


# ------------------------------ Novel Fine-tuning -------------------------------- #
# --> 1. FSRW-like, i.e. run seed0 10 times (the FSOD results on coco in most papers)
for repeat_id in 0 1 2 3 4 5 6 7 8 9
do
    for shot in 10
    do
        for seed in 0
        do
            python3 -m tools.create_config \
                    --dataset coco --config_root configs/coco \
                    --shot ${shot} --seed ${seed} --setting 'fsod'
            CONFIG_PATH=configs/coco/frcn_fsod_r101_novel_${shot}shot_seed${seed}.yaml
            OUTPUT_DIR=${SAVEDIR}/frcn_fsod_r101_novel/fsrw-like/${shot}shot_seed${seed}_repeat${repeat_id}
            python3 -m tools.train_net --num-gpus 8 \
                    --config-file ${CONFIG_PATH} \
                    --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}
            rm ${CONFIG_PATH}
            rm ${OUTPUT_DIR}/model_final.pth
        done
    done
done
python3 tools/extract_results.py --res-dir ${SAVEDIR}/defrcn_fsod_r101_novel/fsrw-like --shot-list 1 2 3 5 10 30  # surmarize all results


# -------------------------------  GFSOD (80 class) ---------------------------------- #
# ----------------------------- Model Preparation --------------------------------- #
python3 -m tools.ckpt_surgery \
        --src ${SAVEDIR}/frcn_r101_base/model_final.pth \
        --method randinit \
        --save-dir ${SAVEDIR}/frcn_r101_base \
        --dataset coco --no-bg
BASE_WEIGHT=${SAVEDIR}/frcn_r101_base/model_reset_surgery_no_bg.pth


# ------------------------------ Novel Fine-tuning ------------------------------- #
# --> 2. TFA-like, i.e. run seed0~9 for robust results (G-FSOD, 80 classes)
for seed in 0 1 2 3 4 5 6 7 8 9
do
    for shot in 10
    do
        python3 -m tools.create_config \
                --dataset coco --config_root configs/coco \
                --shot ${shot} --seed ${seed} --setting 'gfsod'
        CONFIG_PATH=configs/coco/frcn_gfsod_r101_novel_${shot}shot_seed${seed}.yaml
        OUTPUT_DIR=${SAVEDIR}/frcn_gfsod_r101_novel/tfa-like/${shot}shot_seed${seed}
        python3 -m tools.train_net --num-gpus 8 \
                --config-file ${CONFIG_PATH} \
                --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}
        rm ${CONFIG_PATH}
        rm ${OUTPUT_DIR}/model_final.pth
    done
done
python3 -m tools.extract_results \
        --res-dir ${SAVEDIR}/frcn_gfsod_r101_novel/tfa-like \
        --shot-list 10
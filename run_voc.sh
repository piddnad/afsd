#!/usr/bin/env bash

EXPNAME=$1
SAVEDIR=checkpoints/voc/${EXPNAME}
SPLIT_ID=$2

# ------------------------------- Base Pre-train ---------------------------------- #
# Normal Pre-train (2x)
#python3 -m tools.train_net --num-gpus 8 \
#        --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_base_2x.yaml \
#        --opts OUTPUT_DIR ${SAVEDIR}/frcn_r101_base

# for MFA Pre-train
python3 -m tools.train_net --num-gpus 8 \
        --config-file configs/voc/frcn_r101_base${SPLIT_ID}_MFA.yaml \
        --opts OUTPUT_DIR ${SAVEDIR}/frcn_r101_base${SPLIT_ID}


# -------------------------------  GFSOD (80 class) ---------------------------------- #
# ----------------------------- Model Preparation --------------------------------- #
python3 -m tools.ckpt_surgery \
        --src ${SAVEDIR}/frcn_r101_base${SPLIT_ID}/model_final.pth \
        --method randinit \
        --save-dir ${SAVEDIR}/frcn_r101_base${SPLIT_ID} \
        --dataset voc
BASE_WEIGHT=${SAVEDIR}/frcn_r101_base/model_reset_surgery.pth


# ------------------------------ Novel Fine-tuning ------------------------------- #
# --> 2. TFA-like, i.e. run seed0~9 for robust results (G-FSOD, 80 classes)
for seed in 0 # 1 2 3 4 5 6 7 8 9
do
    for shot in 1 2 3 5 10
    do
        python3 -m tools.create_config \
                --dataset voc --config_root configs/voc \
                --shot ${shot} --seed ${seed} --setting 'gfsod'
        CONFIG_PATH=configs/coco/frcn_gfsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
        OUTPUT_DIR=${SAVEDIR}/frcn_gfsod_r101_novel${SPLIT_ID}/tfa-like/${shot}shot_seed${seed}
        python3 -m tools.train_net --num-gpus 8 \
                --config-file ${CONFIG_PATH} \
                --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}
        rm ${CONFIG_PATH}
        rm ${OUTPUT_DIR}/model_final.pth
    done
done
python3 -m tools.extract_results \
        --res-dir ${SAVEDIR}/frcn_gfsod_r101_novel${SPLIT_ID}/tfa-like \
        --shot-list 1 2 3 5 10
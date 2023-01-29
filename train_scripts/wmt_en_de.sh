#!/bin/bash
set -u

GPU=${1}
NUM_GPUS=8
SRC=${2:-'de'}
TGT=${3:-'en'}
LOSS_FUNC="uniform"
LR=0.0001
SEQ_LEN=64
WARMUP=20000
SCHEDULE_UPDATE_STRIDE=20000
UPDATE_GRANU=20
DSET="wmt_ckpts"
INIT_PRETRAINED_MODEL="False"
USE_PRETRAINED_EMBEDDINGS="False"
FREEZE_EMBEDDINGS="False"
LR_ANNEAL_STEPS=1000000
DIFFUSION_STEPS=2000
NOISE_SCHEDULE=sqrt
BATCH_SIZE=128
CHECKPOINT_PATH="ckpts/${DSET}/wmt14_${SEQ_LEN}_${LR}_${DIFFUSION_STEPS}_${LR_ANNEAL_STEPS}_${WARMUP}_schegran${SCHEDULE_UPDATE}_src${SRC}_tgt${TGT}"
TRAIN_TXT_PATH="./data/wmt14_en_de/train"
VAL_TXT_PATH="./data/wmt14_en_de/valid"
IN_CHANNELS=512
WEIGHT_DECAY=0.0
SEED=10708
DROPOUT=0.3
NUM_HEADS=8
CONFIG_NAME="facebook/bart-base"
NOTES="wmt14 training with noise schedule and self condition"

mkdir -p ${CHECKPOINT_PATH}
mkdir -p ${CHECKPOINT_PATH}/log/
export DIFFUSION_BLOB_LOGDIR=${CHECKPOINT_PATH}/log/

ARGS=(--checkpoint_path ${CHECKPOINT_PATH}
    --save_interval ${WARMUP} --lr ${LR}
    --batch_size ${BATCH_SIZE}
    --src ${SRC}
    --tgt ${TGT}
    --diffusion_steps ${DIFFUSION_STEPS}
    --noise_schedule ${NOISE_SCHEDULE}
    --sequence_len ${SEQ_LEN} --seed ${SEED}
    --weight_decay ${WEIGHT_DECAY}
    --predict_xstart True
    --train_txt_path ${TRAIN_TXT_PATH}
    --dataset "wmt14_en_de"
    --val_txt_path ${VAL_TXT_PATH}
    --config_name ${CONFIG_NAME}
    --init_pretrained ${INIT_PRETRAINED_MODEL}
    --freeze_embeddings ${FREEZE_EMBEDDINGS}
    --use_pretrained_embeddings ${USE_PRETRAINED_EMBEDDINGS}
    --notes \""${NOTES}"\")


if [ ${LR_ANNEAL_STEPS} -eq 0 ]; then
    LR_ANNEAL_STEPS=100
    DEBUG=true
else
    DEBUG=false
fi

ARGS+=(--lr_anneal_steps $LR_ANNEAL_STEPS)

if [ $DEBUG = true ]; then
    ARGS+=(--debug)
fi

ARGS+=(--encoder_layers 6
    --decoder_layers 6
    --num_heads 8
    --num_heads 8
    --in_channel 512
    --out_channel 512
    --num_channels 2048
    --sequence_len_src 128
    --warmup $WARMUP
    --schedule_sampler $LOSS_FUNC
    --loss_update_granu $UPDATE_GRANU
    --schedule_update_stride $SCHEDULE_UPDATE_STRIDE)

export CUDA_VISIBLE_DEVICES=$GPU && mpiexec -n $NUM_GPUS python -u main.py "${ARGS[@]}"



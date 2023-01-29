#!/bin/bash

#!/bin/bash


MODEL_NAME=$1
OUT_DIR=${2}
SCHEDULE_PATH=${3}
VAL_TXT=./data/${4}/test
SEED=${5:-10708}

if [ -z "$OUT_DIR" ]; then
    OUT_DIR=${MODEL_NAME}
fi

GEN_BY_Q=${6:-"False"}
GEN_BY_MIX=${7:-"False"}
MIX_PROB=${8:-0}
MIX_PART=${9:-1}
TOP_P=-1
CLAMP="no_clamp"
BATCH_SIZE=100
SEQ_LEN=64
DIFFUSION_STEPS=2000
NUM_SAMPLES=-1

python -u inference_main.py --model_name_or_path ${MODEL_NAME} --sequence_len_src 128 \
--batch_size ${BATCH_SIZE} --num_samples ${NUM_SAMPLES} --top_p ${TOP_P} --time_schedule_path ${SCHEDULE_PATH} \
--seed ${SEED} --val_txt_path ${VAL_TXT} --generate_by_q ${GEN_BY_Q} --generate_by_mix ${GEN_BY_MIX} \
--out_dir ${OUT_DIR} --diffusion_steps ${DIFFUSION_STEPS} --clamp ${CLAMP} --sequence_len ${SEQ_LEN} \
  --generate_by_mix_prob ${MIX_PROB} --generate_by_mix_part ${MIX_PART}


#!/bin/bash

CHECKPOINT_PATH="./ckpt/model_ckpt_001.pt"   
INPUT_PKL_FILE="./examples/5oqv_rb0_msa.pkl.gz"               
OUTPUT_DIR="./results/"                                
ROUNDS=10                                           

python inference.py \
  --checkpoint ${CHECKPOINT_PATH} \
  --input_pkl ${INPUT_PKL_FILE} \
  --ribbon_name 5oqv \
  --output_dir ${OUTPUT_DIR} \
  --rounds ${ROUNDS} \
  --use_dropout true \
  --use_init_structure true

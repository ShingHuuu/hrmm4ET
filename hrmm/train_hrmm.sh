#!/bin/bash

# Select data from {onto, bbn}
export DATA_NAME=bbn
export MODEL_ID=m1
export CUDA_VISIBLE_DEVICES=0

if [[ $DATA_NAME == "onto" ]]; then
    export TRAIN_DATA=onto/onto_train.json
elif [[ $DATA_NAME == "bbn" ]]; then
    export TRAIN_DATA=bbn/bbn_train.json
else
    exit $exit_code
fi

CUDA_VISIBLE_DEVICES=0 python3 -u main.py \
--model_id=$MODEL_ID \
--goal=$DATA_NAME \
--seed=0 \
--train_data=$TRAIN_DATA \
--dev_data=${DATA_NAME}/${DATA_NAME}_dev.json \
--log_period=500 \
--eval_after=500 \
--eval_period=500 \
--alpha_type_reg=0 \
--box_dim=200 \
--box_offset=0.5 \
--emb_type=box \
--gradient_accumulation_steps=16 \
--gumbel_beta=0.006 \
--inv_softplus_temp=1.2471085395024732 \
--learning_rate_cls=0.003720789473794256 \
--learning_rate_enc=2e-05 \
--mode=train \
--model_type=bert-large-uncased-whole-word-masking \
--n_negatives=1000 \
--num_epoch=7 \
--per_gpu_eval_batch_size=8 \
--per_gpu_train_batch_size=8 \
--proj_layer=highway \
--save_period=100000 \
--softplus_scale=1 \
--th_type_vol=0 \
--threshold=0.7 \
--adaptive_thre=False\
--use_gumbel_baysian=True
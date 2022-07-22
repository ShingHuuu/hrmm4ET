#!/bin/bash

# Select data from {onto, bbn}
export DATA_NAME=bbn

CUDA_VISIBLE_DEVICES=0 python3 -u main.py \
--model_id box_${DATA_NAME}_test \
--reload_model_name box_${DATA_NAME}_0605_h_noother/box_${DATA_NAME}_0605_h_noother_best \
--load \
--model_type bert-large-uncased-whole-word-masking \
--mode test \
--constant_density False \
--goal bbn \
--emb_type box \
--threshold 0.7 \
--adaptive_thre False\
--gumbel_beta=0.06 \
--inv_softplus_temp=1.2471085395024732 \
--softplus_scale 1.0 \
--box_dim=200 \
--proj_layer highway \
--per_gpu_eval_batch_size 4 \
--eval_data ${DATA_NAME}/${DATA_NAME}_test.json
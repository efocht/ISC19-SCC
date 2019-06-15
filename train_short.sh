#!/bin/bash

data_dir_train=${1:-./segm_h5_v3_new_split/train/}
data_dir_validation=${2:-./segm_h5_v3_new_split/validation/}

export VE_LOG_LEVEL=0
export OMP_NUM_THREADS=8

python \
    deeplab-tf/deeplab-tf-train.py \
    --datadir_train ${data_dir_train} \
    --datadir_validation ${data_dir_validation} \
    --batch 1 \
    --train_size 10 \
    --validation_size 5 \
    --optimizer opt_type=Adam \
    --decoder bilinear \
    --disable_checkpoints \
    --disable_horovod \
    --data_format channels_first \
    --device /device:VE 
    #--optimizer opt_type=Adam \

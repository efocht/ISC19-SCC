#!/bin/bash

#data_dir_train=${1:-./short_data/train/}
data_dir_train=${1:-./segm_h5_v3_new_split/train/}
#data_dir_validation=${2:-./short_data/validation/}
data_dir_validation=${2:-./segm_h5_v3_new_split/validation/}

#export VE_LOG_LEVEL=3

export LD_PRELOAD=/usr/lib64/libhwloc.so.5
export OMP_NUM_THREADS=2

horovodrun --verbose -np 6 -H localhost:6 python \
    deeplab-tf/deeplab-tf-train.py \
    --datadir_train ${data_dir_train} \
    --datadir_validation ${data_dir_validation} \
    --batch 1 \
    --train_size 24 \
    --validation_size 12 \
    --optimizer opt_type=Adam \
    --decoder bilinear \
    --data_format channels_first \
    --device /device:VE

#!/bin/bash
export CUDA_VISIBLE_DEVICES=3 

if [ $DEBUG == 1 ]; then
    python -m debugpy --listen 5678 --wait-for-client -m manydepth.evaluate_depth \
        --data_path /data/kitti \
        --load_weights_folder log_dir/01_kitti_mr/models/weights_0 \
        --eval_mono
else
    python -m manydepth.evaluate_depth \
        --data_path /data/kitti \
        --load_weights_folder log_dir/01_kitti_mr_semi_supervise/models/weights_17/ \
        --eval_mono
fi
        # --load_weights_folder weights/KITTI_MR \
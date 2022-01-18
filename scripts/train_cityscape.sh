# CUDA_VISIBLE_DEVICES=<your_desired_GPU> \
# python -m manydepth.train \
python -m debugpy --listen 5678 --wait-for-client -m manydepth.train \
    --data_path /data/cityscapes/dump/ \
    --log_dir work_dirs/cityscapes  \
    --model_name cityscape \
    --dataset cityscapes_preprocessed \
    --split cityscapes_preprocessed \
    --freeze_teacher_epoch 5 \
    --height 192 --width 512
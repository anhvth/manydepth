# CUDA_VISIBLE_DEVICES=<your_desired_GPU> \
# python -m debugpy --listen 5678 --wait-for-client -m manydepth.lit_train \
#     --data_path /data/cityscapes/dump/ \
#     --log_dir work_dirs/cityscapes  \
#     --model_name cityscape \
#     --dataset cityscapes_preprocessed \
#     --split cityscapes_preprocessed \
#     --freeze_teacher_epoch 1 \
#     --height 192 --width 512 \
#     --gpus 1


python -m manydepth.lit_train \
    --data_path /data/cityscapes/dump/ \
    --log_dir work_dirs/cityscapes  \
    --model_name cityscape \
    --dataset cityscapes_preprocessed \
    --split cityscapes_preprocessed \
    --freeze_teacher_epoch 5 \
    --height 192 --width 512 \
    --gpus 1
    # --ckpt_path lightning_logs/cityscape/Jan17-23:39:01/ckpts/last.ckpt
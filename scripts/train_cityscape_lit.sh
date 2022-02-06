alias python=/home/av/.conda/envs/manydepth/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7
#DEBUG=1 bash ./scripts/train_cityscape_lit.sh
if [[ $DEBUG == "1" ]]; then 
    echo 'DEBUGING'
    python -m debugpy --listen 5678 --wait-for-client -m manydepth.lit_train \
        --data_path /data/cityscapes/dump/ \
        --log_dir work_dirs/cityscapes  \
        --model_name cityscape \
        --dataset cityscapes_preprocessed \
        --split cityscapes_preprocessed \
        --freeze_teacher_epoch 1 \
        --height 192 --width 512 \
        --gpus 1
else 
    echo 'TRAINING'
    python -m manydepth.lit_train \
        --data_path /data/cityscapes/dump/ \
        --log_dir work_dirs/cityscapes  \
        --model_name cityscape \
        --dataset cityscapes_preprocessed \
        --split cityscapes_preprocessed \
        --freeze_teacher_epoch 5 \
        --height 192 --width 512 \
        --gpus 7
        # --ckpt_path lightning_logs/cityscape/Jan17-23:39:01/ckpts/last.ckpt
fi
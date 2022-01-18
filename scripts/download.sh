DOWNLOADED_PATH=/data/cityscapes
DUMP_ROOT=/data/cityscapes/dump
mkdir -p $DOWNLOADED_PATH
mkdir -p $DUMP_ROOT
cd SfMLearner/data

python prepare_train_data.py \
    --img_height 512 \
    --img_width 1024 \
    --dataset_dir $DOWNLOADED_PATH \
    --dataset_name cityscapes \
    --dump_root $DUMP_ROOT \
    --seq_length 3 \
    --num_threads 40
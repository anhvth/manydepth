export CUDA_VISIBLE_DEVICES=0 
mkdir -p ./log_dir


# ps aux | grep aux 5678 | awk '{print $2}' | xargs kill -9
if [[ $DEBUG==1]]
then 
    python -m debugpy --listen 5678 --wait-for-client -m manydepth.train \
        --data_path /data/kitti \
        --log_dir ./log_dir/  \
        --model_name 01_kitti_mr \
        --num_workers 0
else
    python -m manydepth.train \
        --data_path /data/kitti \
        --log_dir ./log_dir/  \
        --model_name 01_kitti_mr \
        --num_workers 
fi

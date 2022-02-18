WEIGHT=log_dir/01_kitti_mr_semi_supervise/models/weights_11/
python -m manydepth.test_simple \
    --target_image_path outputs/2011_09_26_drive_0001_sync \
    --source_image_path /home/av/gitprojects/manydepth/data/kitti/2011_09_26/2011_09_26_drive_0001_sync/image_02/data \
    --intrinsics_json_path assets/test_sequence_intrinsics.json \
    --model_path $WEIGHT
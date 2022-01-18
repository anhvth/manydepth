# python -m debugpy --listen 5678 --wait-for-client -m manydepth.test_video\
python -m manydepth.test_video \
    --input_dir ~/10s/ \
    --intrinsics_json_path assets/test_sequence_intrinsics.json \
    --model_path CityScapes_MR
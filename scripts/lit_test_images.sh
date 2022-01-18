# python -m debugpy --listen 5678 --wait-for-client -m \
python -m manydepth.lit_test_simple \
    --target_image_path assets/germany15s/00001.jpg \
    --source_image_path assets/germany15s/00000.jpg \
    --intrinsics_json_path assets/test_sequence_intrinsics.json \
    --model_path CityScapes_MR
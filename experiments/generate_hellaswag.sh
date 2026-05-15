# Default
python -m utils.generate_hellaswag \
    --model_name_or_path "HuggingFaceH4/zephyr-7b-beta" \
    --output_dir "checkpoints/zephyr-7b-beta-hellaswag"


# in French
python -m utils.generate_hellaswag \
    --model_name_or_path "HuggingFaceH4/zephyr-7b-beta" \
    --language French \
    --output_dir "checkpoints/zephyr-7b-beta-hellaswag"

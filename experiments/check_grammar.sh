# Generate responses for RAd models
models=(
    rm/rad/random/HuggingFaceH4/zephyr-7b-beta_alpha-1200-1200_coeffs-20.0-20.0_batches-500_layer-7_component-None_nu-0.0
    rm/rad/truth/HuggingFaceH4/zephyr-7b-beta_alpha-1200-1200_coeffs-14.0-14.0_batches-500_layer-7_component-None_nu-0.0
    rm/rad/sentiment/negative_direction/HuggingFaceH4/zephyr-7b-beta_alpha-1200-1200_coeffs-23.0-23.0_batches-500_layer-7_component-None_nu-0.0
    rm/rad/refusal/refusal_direction/HuggingFaceH4/zephyr-7b-beta_alpha-1200-1200_coeffs-18.0-18.0_batches-500_layer-7_component-None_nu-0.0
)

for model in "${models[@]}"; do
    output_dir="checkpoints/wmdp_generate/${model}"

    if [ -d "$output_dir" ]; then
        echo "Directory $output_dir already exists. Skipping..."
        continue
    fi

    rm -rf "$output_dir"
    mkdir -p "$output_dir"

    python -m utils.generate_qa.generate_qa \
        --model_name_or_path "checkpoints/${model}" \
        --output_dir "$output_dir"
done


# Grammar check: RAd (random) vs. RAd (truth, sentiment, refusal) on WMDP
anchor="rm/rad/random/HuggingFaceH4/zephyr-7b-beta_alpha-1200-1200_coeffs-20.0-20.0_batches-500_layer-7_component-None_nu-0.0"
competitors=(
    rm/rad/truth/HuggingFaceH4/zephyr-7b-beta_alpha-1200-1200_coeffs-14.0-14.0_batches-500_layer-7_component-None_nu-0.0
    rm/rad/sentiment/negative_direction/HuggingFaceH4/zephyr-7b-beta_alpha-1200-1200_coeffs-23.0-23.0_batches-500_layer-7_component-None_nu-0.0
    rm/rad/refusal/refusal_direction/HuggingFaceH4/zephyr-7b-beta_alpha-1200-1200_coeffs-18.0-18.0_batches-500_layer-7_component-None_nu-0.0
)

python -m utils.generate_qa.check_grammar \
    --anchor "$anchor" \
    --competitors "${competitors[@]}" \
    --output_dir "checkpoints/wmdp_generate"


# Extract win/tie rates
python -m utils.generate_qa.extract_win_rates

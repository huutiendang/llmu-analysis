source $(conda info --base)/etc/profile.d/conda.sh
conda activate safety-unlearning


export WANDB_PROJECT="Beyond-Forgetting"
export WANDB_LOG_MODEL="false"


# RAd random and truth directions
models=(checkpoints/rm/rad/random_direction/HuggingFaceH4/zephyr-7b-beta_alpha-1200-1200_coeffs-14.0-14.0_batches-500_layer-7_component-None_nu-0.0
        checkpoints/rm/rad/truth/truth_direction/HuggingFaceH4/zephyr-7b-beta_alpha-1200-1200_coeffs-14.0-14.0_batches-500_layer-7_component-None_nu-0.0)

# RAb random and truth directions
models=(checkpoints/rm/rab/random_direction/HuggingFaceH4/zephyr-7b-beta_alpha-20-20_coeffs-50.0-50.0_batches-500_layer-7_component-None_nu-0.0
        checkpoints/rm/rab/truth/truth_direction/HuggingFaceH4/zephyr-7b-beta_alpha-20-20_coeffs-50.0-50.0_batches-500_layer-7_component-None_nu-0.0)

# RAd and RAb sentiment positive
models=(checkpoints/rm/rad/sentiment/positive_direction/HuggingFaceH4/zephyr-7b-beta_alpha-1200-1200_coeffs-16.0-16.0_batches-500_layer-7_component-None_nu-0.0
        checkpoints/rm/rab/sentiment/positive_direction/HuggingFaceH4/zephyr-7b-beta_alpha-20-20_coeffs-120.0-120.0_batches-500_layer-7_component-None_nu-0.0)

# RAd and RAb refusal
models=(checkpoints/rm/rad/refusal/refusal_direction/HuggingFaceH4/zephyr-7b-beta_alpha-1200-1200_coeffs-18.0-18.0_batches-500_layer-7_component-None_nu-0.0
        checkpoints/rm/rab/refusal/refusal_direction/HuggingFaceH4/zephyr-7b-beta_alpha-20-20_coeffs-40.0-40.0_batches-500_layer-7_component-None_nu-0.0)


for unlearned_model in "${models[@]}"
do
    for n_samples in 5 10 50 100 500 1000
    do
        for dataset in wmdp_bio-forget-corpus wmdp_bio-retain-corpus wmdp_cyber-forget-corpus wmdp_cyber-retain-corpus wikitext
        do 
            python -m src.finetuning.finetune \
                --model $unlearned_model \
                --dataset $dataset \
                --n_samples $n_samples
        done
    done
done

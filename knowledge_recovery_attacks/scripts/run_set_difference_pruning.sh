cd src/set_difference_pruning

source $(conda info --base)/etc/profile.d/conda.sh
conda activate prune_llm


models=("zephyr-7b-beta_rad.random"
        "zephyr-7b-beta_rad.truth"
        "zephyr-7b-beta_rad.sentiment.positive"
        "zephyr-7b-beta_rad.refusal"
        "zephyr-7b-beta_rab.random"
        "zephyr-7b-beta_rab.truth"
        "zephyr-7b-beta_rab.sentiment.positive"
        "zephyr-7b-beta_rab.refusal")


for model in "${models[@]}"
do
    for subset in wmdp-bio wmdp-cyber
    do
        method="wandg"
        type="unstructured"
        for prune_data in "$subset-forget" wikitext; do
            save_dir="../../results/pruning/data/$model/$type/$method/$prune_data/"
            
            if [ -d $save_dir ]; then
                continue
            fi

            mkdir -p $save_dir
            python -m main \
                --model $model \
                --prune_method $method \
                --prune_data $prune_data \
                --nsamples 128 \
                --sparsity_ratio 0.5 \
                --sparsity_type $type \
                --save $save_dir \
                --dump_wanda_score
        done


        method="wandg_set_difference"
        save_dir="../../results/pruning/data/$model/$type/$method"
        for p in 0.005 0.01 0.025 0.05 0.075; do
            for q in 0.005 0.01 0.025 0.05 0.075; do
                if (( $(echo "$p < $q" | bc -l) ))
                then
                    continue
                fi

                python -m main \
                    --model $model \
                    --prune_method $method \
                    --sparsity_ratio 0.5 \
                    --prune_data $subset-forget \
                    --p $p \
                    --q $q \
                    --sparsity_type $type \
                    --save $save_dir \
                    --save_model $save_dir

                mkdir -p "../../results/pruning/results/$model-$subset/p_${p}_q_${q}"
                lm_eval --model hf \
                    --model_args "pretrained=$save_dir" \
                    --tasks wmdp_bio,wmdp_cyber,mmlu \
                    --output_path "../../results/pruning/results/$model-$subset/p_${p}_q_${q}"
            done
        done
    done
done

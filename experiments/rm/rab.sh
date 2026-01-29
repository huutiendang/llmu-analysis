for SEED in 42
do
    for ALPHA in 20
    do
        for DIRECTION in random
        do
            for COEFFS in 60
            do
                for BATCH in 500
                do
                    for ID in 7
                    do
                        for NU in 0.0
                        do
                            python -m baselines.rm.rab.unlearn \
                                --model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
                                --max_num_batches $BATCH \
                                --alpha "${ALPHA},${ALPHA}" \
                                --direction $DIRECTION \
                                --steering_coeffs "${COEFFS},${COEFFS}" \
                                --seed $SEED \
                                --batch_size 4 \
                                --nu $NU \
                                --layer_id $ID \
                                --layer_ids "$((ID - 2)),$((ID - 1)),$ID" \
                                --verbose;
                        done
                    done
                done
            done
        done
    done
done

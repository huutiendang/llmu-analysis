for SEED in 42
do
    for ALPHA in 1200
    do
        for DIRECTION in refusal
        do
            for COEFFS in 20
            do
                for BATCH in 500
                do
                    for ID in 7
                    do
                        for NU in 0.0 #0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1
                        do
                            python -m baselines.rm.rad.unlearn \
                                --model_name_or_path "HuggingFaceH4/zephyr-7b-beta" \
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

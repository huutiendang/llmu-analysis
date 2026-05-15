# Extract concept vectors with diff-in-means, K-Means, and Ridge Regression
python -m utils.vary_probes \
    --model "HuggingFaceH4/zephyr-7b-beta" \
    --layer 7

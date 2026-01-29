# Knowledge Recovery Attacks
The code is based on that of [An Adversarial Perspective on Machine Unlearning for AI Safety](https://arxiv.org/abs/2409.18025) by Jakub Łucki, Boyi Wei, Yangsibo Huang, Peter Henderson, Florian Tramèr, and Javier Rando.

## Setup
There are two separate environments:

For pruning, install the conda environment `prune_llm` with:
```bash
cd knowledge_recovery_attacks/src/set_difference_pruning
conda env create -f env_prune_llm.yml
```

For the other attacks, install the conda environment `safety-unlearning` with:
```bash
conda env create -f env.yml
```

Further, to track your experiments, set your Wandb access token via:
```bash
export WANDB_API_KEY="your_wandb_api_key"
```

## Running experiments

All code are provided in `scr/`. Ultimately, we provide scripts for reproducing the main results in `scripts/`. 

An example of running the Logitlens attack is:
```bash
cd knowledge_recovery_attacks
bash scripts/run_logit_lens.sh
```
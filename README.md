# llmu-analysis

Utilities and experiments for machine unlearning analysis, including baseline evaluations, RM (RAB/RAD) unlearning methods, QA generation/grammar checks, and knowledge recovery attacks.

## What’s inside
- `baselines/`: evaluation baselines (factual QA, GSM8K, HellaSwag, refusal, TruthfulQA) and RM unlearning methods.
- `experiments/`: ready-to-run scripts for common workflows.
- `utils/`: QA generation, probes, grammar checks, and helpers.
- `knowledge_recovery_attacks/`: attack implementations and scripts.
- `outputs/`: collected results from experiments.

## Quickstart

### 1) Create environments

**Knowledge Recovery Attacks (general):**
```bash
cd knowledge_recovery_attacks
conda env create -f [env.yml](http://_vscodecontentref_/4)
```

**Set-Difference Pruning (separate env):**

```bash
cd knowledge_recovery_attacks/src/set_difference_pruning
conda env create -f env_prune_llm.yml
```

## Running Baseline/Utility Scripts
All commands below are from the repo root:

Generate HellaSwag responses:
```bash
bash [generate_hellaswag.sh](http://_vscodecontentref_/5)
```

Extract concept vectors (probes):
```bash
bash [vary_probes.sh](http://_vscodecontentref_/6)
```

Generate QA + grammar checks (WMDP):
```bash
bash [check_grammar.sh](http://_vscodecontentref_/7)
```

Run RAD unlearning (example sweep):
```bash
bash [rad.sh](http://_vscodecontentref_/8)
```

Knowledge Recovery Attacks

All attack code and scripts are under knowledge_recovery_attacks/.

Example: Logit Lens attack
```bash
cd knowledge_recovery_attacks
bash scripts/run_logit_lens.sh
```

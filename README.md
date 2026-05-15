# Beyond Forgetting: Machine Unlearning Elicits Controllable Side Behaviors and Capabilities

## Abstract

*We consider Representation Misdirection (RM), a class of large language model (LLM) unlearning methods that achieve forgetting by redirecting the forget-representations, that is, latent representations of forget-samples, toward a target vector. Despite being important, the roles of the target vector used in RM, however, remain underexplored. Here, we approach and revisit RM through the lens of the Linear Representation Hypothesis. Specifically, if one can identify a one-dimensional representation corresponding to a high-level concept, the Linear Representation Hypothesis enables linear operations on this concept vector within the forget-representation space. Under this view, we hypothesize that, beyond forgetting, machine unlearning via RM elicits controllable emergent side behaviors and stronger side capabilities corresponding to the high-level concept. Our hypothesis is empirically validated across a wide range of tasks, including behavioral control (e.g., controlling unlearned models' truthfulness, sentiment, refusal, and language) and capability enhancement (e.g., improving unlearned models' in-context learning (ICL) capability). Our findings reveal that this phenomenon could be either a hidden risk if misused or a mechanism that can be harnessed for developing unlearned models that require stronger capabilities and controllable behaviors.*

## What’s inside

- [`baselines/`](./baselines/): implementations of RAd and RAb & concept vector extraction.
- [`knowledge_recovery_attacks/`](./knowledge_recovery_attacks/): implementations and scripts of knowledge recovery attacks.
- [`utils/`](./utils/): implementations of language control and models' outputs analysis.
- [`experiments/`](./experiments/): scripts for running experiments.

## Setup

To set up the environment for Unlearning, run:
```bash
conda create -n unlearning python=3.13
conda activate unlearning
pip install -r requirements.txt
```

For Knowledge Recovery Attacks, refer to [knowledge_recovery_attacks/](./knowledge_recovery_attacks/).

## Running experiments

Refer to [experiments/](./experiments/) for scripts to run experiments.

An example of running RAd w/ truth:
```bash
bash experiments/extract_truth_vector.sh
bash experiments/rm/rad.sh
```

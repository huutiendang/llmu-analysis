import re
import json
import pandas as pd
from pathlib import Path


def load_anchor_competitors(anchor, task, base_dir="checkpoints/wmdp_generate/qa_grammar_check"):
    root = Path(base_dir)
    anchor = anchor.replace("/", "__")

    anchor_dirs = [p for p in root.iterdir() if p.is_dir() and f"anchor:{anchor}" in p.name]
    if not anchor_dirs:
        raise FileNotFoundError(f"No anchor folder for {anchor} in {base_dir}")

    dfs = []
    for anchor_dir in anchor_dirs:
        for path in anchor_dir.glob(f"**/{task}.jsonl"):
            if "base_model" in path._str:
                continue

            df = pd.read_json(path, lines=True)
            df["competitor"] = path.parent.name
            df["anchor"] = anchor
            df["source_file"] = str(path)
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def parse_embedded_json(text):
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        raise ValueError("No JSON object found")
    return json.loads(m.group(0))


def get_verdict(text, response_anchor_first):
	try:
		data = parse_embedded_json(text)
		verdict = data.get("verdict", None)

		if response_anchor_first:
			who_wins_mapping = {
				"[[A]]": "anchor wins",
				"[[B]]": "competitor wins",
                "[[C]]": "tie",           
			}
		else:
			who_wins_mapping = {
				"[[A]]": "competitor wins",
				"[[B]]": "anchor wins",
                "[[C]]": "tie",           
			}

		return who_wins_mapping.get(verdict, None)

	except ValueError:
		return None


if __name__ == "__main__":
    method_labels = {
        "base_model": "Base model",

        "rm/rmu": "RMU",

        "rm/rad/random": "RAd (rand)",
        "rm/rad/truth": "RAd (truth)",
        "rm/rad/sentiment/negative_direction": "RAd (neg→pos)",
        "rm/rad/sentiment/positive_direction": "RAd (pos→neg)",
        "rm/rad/refusal/refusal_direction": "RAd (refusal)",

        "rm/rad/factual/linguistic/antonyms_direction": "RAd (antonym)",
        "rm/rad/factual/linguistic/present2past_direction": "RAd (pres→past)",
        "rm/rad/factual/knowledge/country2capital_direction": "RAd (ctry→cap)",
        "rm/rad/factual/knowledge/person2language_direction": "RAd (pers→lang)",

        "rm/rab/random": "RAb (rand)",
        "rm/rab/truth/truth_direction": "RAb (truth)",
        "rm/rab/sentiment/negative_direction": "RAb (neg→pos)",
        "rm/rab/sentiment/positive_direction": "RAb (pos→neg)",
        "rm/rab/refusal/refusal_direction": "RAb (refusal)",

        "ga/ga_kl": "GA+KL",
        "ga/ga_mse": "GA+MSE",
        "dpo/dpo_kl": "DPO+KL",
        "dpo/dpo_mse": "DPO+MSE",
        "npo/npo_kl": "NPO+KL",
        "npo/npo_mse": "NPO+MSE",
        "sim_npo/sim_npo_kl": "SimNPO+KL",
        "sim_npo/sim_npo_mse": "SimNPO+MSE",
    }

    dfs = []
    for task in ["wmdp-bio", "wmdp-cyber", "merged_samples_mmlu_all_subjects"]:
        for anchor in ["rm/rad/random", "rm/rad/truth"]:
            try:
                dataset_name = "wmdp" if "wmdp" in task else "mmlu"
                competitor_df = load_anchor_competitors(anchor, task, base_dir=f"checkpoints/{dataset_name}_generate/qa_grammar_check")
                competitor_df["verdict"] = competitor_df.apply(lambda row: get_verdict(row["Qwen2.5-32B-Instruct-GPTQ-Int4"], row["response_anchor_first"]), axis=1)

                print("task:", task, "; anchor:", anchor, "; invalid verdict ratio:", competitor_df["verdict"].apply(lambda x: x is None).mean())

                # Count wins/losses for each competitor
                competitor_df = competitor_df.groupby("competitor")["verdict"].value_counts().to_frame().unstack()

                # Normalize the format
                competitor_df.columns = competitor_df.columns.droplevel(0)
                competitor_df.columns.name = None
                competitor_df = competitor_df.reset_index()

                # Calculate percentages, change competitor names, and sort by "anchor wins"

                verdict_sum = competitor_df[["anchor wins", "competitor wins", "tie"]].sum(axis=1)
                competitor_df["anchor wins"] = competitor_df["anchor wins"] / verdict_sum * 100
                competitor_df["competitor wins"] = competitor_df["competitor wins"] / verdict_sum * 100
                competitor_df["tie"] = competitor_df["tie"] / verdict_sum * 100

                competitor_df["competitor"] = competitor_df["competitor"].apply(lambda x: x.replace("competitor:", "")\
                                                                                        .split("__zephyr-7b-beta")[0]\
                                                                                        .split("__HuggingFaceH4")[0]\
                                                                                        .replace("__", "/"))
                competitor_df["competitor"] = competitor_df["competitor"].map(method_labels)
                competitor_df["anchor"] = method_labels[anchor]

                competitor_df["task"] = task
                
                # Sort
                competitors_order = [
                    "RAd (truth)",
                    "RAd (neg→pos)",
                    "RAd (pos→neg)",
                    "RAd (refusal)",

                    "RAd (antonym)",
                    "RAd (pres→past)",
                    "RAd (ctry→cap)",
                    "RAd (pers→lang)",

                    "RAb (truth)",
                    "RAb (pos→neg)",
                    "RAb (neg→pos)",
                    "RAb (refusal)",

                    "GA+KL",
                    "GA+MSE",
                    "DPO+KL",
                    "DPO+MSE",
                    "NPO+KL",
                    "NPO+MSE",
                    "SimNPO+KL",
                    "SimNPO+MSE",
                    "RMU"
                ]

                competitor_df["competitor"] = pd.Categorical(
                    competitor_df["competitor"],
                    categories=competitors_order,
                    ordered=True
                )

                competitor_df = competitor_df.sort_values("competitor").reset_index(drop=True)

                dfs.append(competitor_df[["anchor", "anchor wins", "tie", "competitor wins", "competitor", "task"]])
            
            except:
                pass

    df = pd.concat(dfs, ignore_index=True)
    df.to_json("checkpoints/qa_grammar_check/summary_win_rates.jsonl", orient="records", lines=True)

    print("Summary saved to checkpoints/qa_grammar_check/summary_win_rates.jsonl")

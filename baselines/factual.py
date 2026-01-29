import os
import json
import numpy as np
import torch
from datasets import load_dataset
import tqdm as tqdm
import pickle
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from utils import *

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import pandas as pd

TASK_CONFIG = {
    "country2capital": {
        "data_path": "data/knowledge/country_capital.json",
        "zeroshot_template": "Text: {input}\nAnswer:",
        "instruction_template": "Text: {input}\nCapital:",
        # "instruction_template": "The following input is the name of a country. Determine its capital. Output only the capital name.\nText: {input}\nAnswer: {label}",
        "task_type": "knowledge"
    },
    "person2language": {
        "data_path": "data/knowledge/person_language.json",
        "zeroshot_template": "Text: {input}\nAnswer:",
        "instruction_template": "Text: {input}\nLanguage:",
        # "instruction_template": "The following input is the name of a person. Identify the language they are associated with (e.g., native language or primary language used in their works). Output only one language.\n{input} -> {label}",
        "task_type": "knowledge"
    },
    "antonyms": {
        "data_path": "data/linguistic/antonyms.json",
        "zeroshot_template": "Text: {input}\nAnswer:",
        "instruction_template": "Text: {input}\nAntonym:",
        # "instruction_template": "Given a word, output its antonym. Output with only one word.\n{input} -> {label}",
        "task_type": "linguistic"
    },
    "present2gerund": {
        "data_path": "data/linguistic/present_simple_gerund.json",
        "zeroshot_template": "Text: {input}\nAnswer:",
        "instruction_template": "Text: {input}\nGerund:",
        # "instruction_template": "Given an English verb in its base form. Identify its gerund form. Output with only one word.\n{input} -> {label}",
        "task_type": "linguistic"
    },
    "present2past": {
        "data_path": "data/linguistic/present_simple_past_simple.json",
        "zeroshot_template": "Text: {input}\nAnswer:",
        "instruction_template": "Text: {input}\nPast simple:",
        # "instruction_template": "Given an English verb in its base form. Identify its past simple form. Output with only one word.\n{input} -> {label}",
        "task_type": "linguistic"
    },
    "en2fr": {
        "data_path": "data/translation/en_fr.json",
        "zeroshot_template": "Text: {input}\nAnswer:",
        "instruction_template": "Text: {input}\nFrench:",
        "task_type": "translation"
    },
    "en2es": {
        "data_path": "data/translation/en_es.json",
        "zeroshot_template": "Text: {input}\nAnswer:",
        "instruction_template": "Text: {input}\nSpanish:",
        "task_type": "translation"
    },
    "sst2": {
        "data_path": "data/translation/en_es.json",
        "zeroshot_template": "Text: {input}\nAnswer:",
        "instruction_template": "Text: {input}\nSentiment:",
        "task_type": "sentiment"
    }
}

class OutputProcess:
    @staticmethod
    def extract_capital(raw_pred, ground_truth):
        raw_pred = raw_pred.strip()
    
        raw_pred = raw_pred.split('\n')[0].strip()
    
        raw_pred = re.sub(r'[.,;!?]+$', '', raw_pred)
        raw_pred = raw_pred.strip('"\'')

        # Normalize for comparison
        norm_pred = OutputProcess.normalize_text(raw_pred)
        norm_gt = OutputProcess.normalize_text(ground_truth)
        
        # Strategy 1: Exact match (normalized)
        if norm_pred == norm_gt:
            return ground_truth 
        
        if norm_gt in norm_pred:
            return ground_truth  # Return ground truth for consistency
        
        gt_words = set(norm_gt.split())
        pred_words = set(norm_pred.split())
        
        if gt_words and gt_words.issubset(pred_words):
            return ground_truth 

        return raw_pred if raw_pred else None

    @staticmethod
    def normalize_text(text):
        """Normalize text for comparison (lowercase, remove extra spaces)"""
        if text is None:
            return ""
        return ' '.join(text.lower().strip().split())
    @staticmethod
    def exact_match(pred, gt):
        assert len(pred) == len(gt)
        return sum(
            p.strip().lower() == g.strip().lower()
            for p, g in zip(pred, gt)
        ) / len(gt)
        
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_target_module(model, layer_id: int, module_template: str, sub_component: str | None = None):
    # Split path into components
    path = module_template.format(model_name="model", layer_id=layer_id)
    parts = path.split(".")[1:]  # remove model_name
    module = model

    for part in parts:
        if "[" in part and "]" in part:
            name, idx = part.split("[")
            idx = int(idx.strip("]"))
            module = getattr(module, name)[idx]
        else:
            module = getattr(module, part)

    if sub_component and sub_component != "None":
        for subpart in sub_component.split("."):
            module = getattr(module, subpart)

    return module

def load_model(model_name_or_path):
    torch_dtype = "auto" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, 
        trust_remote_code=True, 
        use_fast=False,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    return model, tokenizer

def predict(model, tokenizer, input_text, max_new_tokens, device="cuda"):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    out = tokenizer.decode(outputs[0])
    return out[len(input_text):]


def train_val_test_split(data, ratios=(4, 1, 5), seed=42):
    random.seed(seed)
    keys = list(data.keys())
    random.shuffle(keys)

    n = len(keys)
    r1, r2, r3 = ratios
    total = r1 + r2 + r3

    n1 = int(n * r1 / total)
    n2 = int(n * r2 / total)

    train_keys = keys[:n1]
    val_keys = keys[n1:n1 + n2]
    test_keys = keys[n1 + n2:]

    train = {k: data[k] for k in train_keys}
    val = {k: data[k] for k in val_keys}
    test = {k: data[k] for k in test_keys}

    return train, val, test

def tokenize_dataset(data: List[str], tokenizer, template: str) -> List:
    all_prompts = []
    for input_text, label in data.items():
        prompt = template.format(input=input_text)
        tokenized = tokenizer(prompt, return_tensors='pt')
        all_prompts.append(tokenized)
    return all_prompts


def load_task_data(task_name: str) -> Tuple[Dict, Dict]:
    if task_name not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(TASK_CONFIG.keys())}")
    
    config = TASK_CONFIG[task_name]
    data = load_json(config["data_path"])
    
    return data, config

def prepare_training_data(data: Dict, tokenizer, config: Dict):
    # keys = list(data_split.keys())
    
    # Zeroshot prompts (label 0)
    zeroshot_inputs = tokenize_dataset(data, tokenizer, config["zeroshot_template"])
    zeroshot_labels = [0] * len(data)
    
    # Instruction prompts (label 1)
    instruction_inputs = tokenize_dataset(data, tokenizer, config["instruction_template"])
    instruction_labels = [1] * len(data)
    
    # Combine
    all_inputs = zeroshot_inputs + instruction_inputs
    all_labels = zeroshot_labels + instruction_labels
    
    return all_inputs, all_labels

def extract_activations(model, inputs_list: List, hook_module) -> List:
    activations = []
    for inputs in tqdm.tqdm(inputs_list):
        acts = forward_with_cache(
            model, inputs, module=hook_module, no_grad=False
        ).mean(dim=1).detach().cpu()
        activations.append(acts)
    return activations


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--task", type=str, default="person2language", 
                        choices=list(TASK_CONFIG.keys()),
                        help="Task to train on")
    parser.add_argument("--layer_id", type=int, default=7, help="Layer to extract activations from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    return args 

if __name__ == "__main__":
    args = get_args()

    SEED = args.seed
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model, tokenizer = load_model(args.model_name_or_path)
    data, config = load_task_data(args.task)
    
    train, val, test = train_val_test_split(data, seed=SEED)
    
    train_inputs, train_labels = prepare_training_data(train, tokenizer, config)
    val_inputs, val_labels = prepare_training_data(val, tokenizer, config)

    hook_module = get_target_module(
        model=model, 
        layer_id=args.layer_id, 
        module_template="{model_name}.model.layers[{layer_id}]", 
        sub_component=None
    )
    
    train_activations = extract_activations(model, train_inputs, hook_module)
    val_activations = extract_activations(model, val_inputs, hook_module)
    
    x_train = torch.cat(train_activations, dim=0).to(torch.float32).numpy()
    y_train = np.array(train_labels)
    x_val = torch.cat(val_activations, dim=0).to(torch.float32).numpy()
    y_val = np.array(val_labels)
    # train cls
    clf = LogisticRegression(random_state=SEED, max_iter=5000)
    clf.fit(x_train, y_train)
    # test on validation
    y_val_pred = clf.predict(x_val)
    acc = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {acc:.4f}")
    save_path = f"data/target_directions/lp/instruction/{config['task_type']}/{args.task}/mistral_{args.task}_direction_probe_layer-{args.layer_id}.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_probes(clf, path=save_path)
    print()
    

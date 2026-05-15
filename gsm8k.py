import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import sys
import pickle
from baselines.utils import *
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.cluster import KMeans

def format_gsm8k(question):
    return f"Question: {question}\nAnswer:"

def format_gsm8k_cot_zeroshot(question): 
    return f"Q: {question}\nA: Let's think step by step."

def format_gsm8k_fewshot(question, fewshot_samples):
    prompt_template = "Question: {question}\nAnswer:"

    samples = []
    for i in range(len(fewshot_samples)):
        fewshot_question = fewshot_samples.loc[i, "question"]
        fewshot_answer = fewshot_samples.loc[i, "answer"]
        fewshot_prompt = prompt_template.format(question=fewshot_question) + " " + fewshot_answer
        samples.append(fewshot_prompt)

    return "\n\n".join(samples) + "\n\n" + prompt_template.format(question=question)


def tokenized_gsm8k(dataset, tokenizer, num_fewshot=0):
    if num_fewshot != 0:
        train_gsm8k = load_dataset("openai/gsm8k", "main", split="train")
        train_gsm8k = train_gsm8k.train_test_split(test_size=0.2, seed=42)["train"]
        train_gsm8k = train_gsm8k.shuffle(seed=42).select(range(num_fewshot))
        fewshot_samples = train_gsm8k.to_pandas()

    all_prompts = []
    all_labels = []
    for i in range(len(dataset)): 
        question = dataset[i]['question']

        # gsm8k
        prompt = format_gsm8k(question)
        prompt = tokenizer(prompt, return_tensors = 'pt')
        all_prompts.append(prompt)
        all_labels.append(0)

        # gsm8k_cot_zeroshot/gsm8k_fewshot
        if num_fewshot == 0:
            prompt = format_gsm8k_cot_zeroshot(question)
        else:
            prompt = format_gsm8k_fewshot(question, fewshot_samples)
        prompt = tokenizer(prompt, return_tensors = 'pt')
        all_prompts.append(prompt)
        all_labels.append(1)
        
    return all_prompts, all_labels

def find_direction_logistic_regression(x_train, y_train):
    clf = LogisticRegression(random_state=42, max_iter=1000).fit(x_train, y_train)
    return clf

def find_direction_mean_difference(x_train, y_train):
    x_class_0 = x_train[y_train == 0]
    x_class_1 = x_train[y_train == 1]
    
    mean_class_0 = np.mean(x_class_0, axis=0)
    mean_class_1 = np.mean(x_class_1, axis=0)

    mean_diff = mean_class_1 - mean_class_0

    return mean_diff



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

if __name__ == "__main__":
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.train_test_split(test_size=0.2, seed=42)
    d_train, d_val = ds["train"], ds["test"]

    model_name_or_path = "mistralai/Mistral-7B-v0.1"
    model, tokenizer = load_model(model_name_or_path,)
    
    # train_size = 300
    # d_train = d_train.train_test_split(train_size=train_size, seed=42)["train"]
    
    num_fewshot = 0  # 0 for gsm8k_cot_zeroshot, otherwise gsm8k_fewshot
    train_inputs, train_labels = tokenized_gsm8k(d_train, tokenizer, num_fewshot=num_fewshot)
    val_inputs, val_labels = tokenized_gsm8k(d_val, tokenizer, num_fewshot=num_fewshot)

    layer_id = 7

    model.config.num_hidden_layers = layer_id + 1
    
    hook_module = get_target_module(
        model=model, layer_id=layer_id, module_template="{model_name}.model.layers[{layer_id}]", sub_component=None
    )
    
    train_activations, val_activations = [], []
    
    for inputs in tqdm.tqdm(train_inputs):
        inputs.to(model.device)
        activations = forward_with_cache(
            model, inputs, module=hook_module, no_grad=False
        ).mean(dim=1).detach().cpu()
        train_activations.append(activations)
    
    x_train = torch.cat(train_activations, dim=0).to(torch.float32).numpy()
    y_train = np.array(train_labels)
    
    # direction_md = find_direction_mean_difference(x_train, y_train)
    
    # save_probes(direction_md, path=f"data/target_directions/md/truth/truth_direction_probe_layer-{layer_id}.pkl")
    
    for inputs in tqdm.tqdm(val_inputs):
        inputs.to(model.device)
        activations = forward_with_cache(
            model, inputs, module=hook_module, no_grad=False
        ).mean(dim=1).detach().cpu()
        val_activations.append(activations)
    
# 
    x_val = torch.cat(val_activations, dim=0).to(torch.float32).numpy()
    y_val = np.array(val_labels)
    # train cls
    clf = LogisticRegression(random_state=42, max_iter=1000).fit(x_train, y_train)
    
    # test on validation
    y_val_pred = clf.predict(x_val)
    acc = accuracy_score(y_val, y_val_pred)
    
    # zephyr layer 15 acc 78.1, mistral layer 15 acc 76.8 

    # zephyr layer 7, probe training size 10,   acc 56.3
    # zephyr layer 7, probe training size 50,   acc 69.8
    # zephyr layer 7, probe training size 100,  acc 70.0
    # zephyr layer 7, probe training size 200,  acc 72.0
    # zephyr layer 7, probe training size 300,  acc 71.3
    # zephyr layer 7, probe training size full, acc 71.3

    # zephyr layer 7, LogisticRegression probe on gsm8k, acc 99.9
    # zephyr layer 7, LogisticRegression probe on gsm8k_5shot, acc 100.0

    # muse-news layer 7 acc 73.2
    # muse-books layer 7 acc 74.2

    # mistral-v0.1 layer 7, LogisticRegression probe on gsm8k, acc 99.9

    
    if num_fewshot == 0:
        concept_name = "gsm8k"
    else:        
        concept_name = f"gsm8k_{num_fewshot}shot"
    
    save_probes(clf, path=f"data/target_directions/lp/{concept_name}/{model_name_or_path}_layer-{layer_id}.pkl")
    print(acc)

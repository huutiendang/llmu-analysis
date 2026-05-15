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

def format_hellaswag(context, ):
    return f"Finish this sentence: {context} Answer:"

def format_hellaswag_in_language(context, language):
    return f"Finish this sentence: {context} Answer: in {language}:"

def tokenized_hellaswag(dataset, tokenizer, language):
    all_prompts = []
    all_labels = []
    for i in range(len(dataset)): 
        context = dataset[i]['ctx']

        # hellaswag
        prompt = format_hellaswag(context)
        prompt = tokenizer(prompt, return_tensors = 'pt')
        all_prompts.append(prompt)
        all_labels.append(0)

        # hellaswag_in_language
        prompt = format_hellaswag_in_language(context, language)
        prompt = tokenizer(prompt, return_tensors = 'pt')
        all_prompts.append(prompt)
        all_labels.append(1)

    return all_prompts, all_labels


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
    ds = load_dataset("Rowan/hellaswag", split="train")
    ds = ds.train_test_split(test_size=0.2, seed=42)
    d_train, d_val = ds["train"], ds["test"]

    to_language = "Japanese"
    model_name_or_path = "HuggingFaceH4/zephyr-7b-beta"
    model, tokenizer = load_model(model_name_or_path,)
    
    print(f"Extracting probes for {model_name_or_path} on hellaswag with language {to_language}...")
    train_inputs, train_labels = tokenized_hellaswag(d_train, tokenizer, language=to_language)
    val_inputs, val_labels = tokenized_hellaswag(d_val, tokenizer, language=to_language)

    layer_id = 7
    
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
    
    for inputs in tqdm.tqdm(val_inputs):
        inputs.to(model.device)
        activations = forward_with_cache(
            model, inputs, module=hook_module, no_grad=False
        ).mean(dim=1).detach().cpu()
        val_activations.append(activations)
    
    x_val = torch.cat(val_activations, dim=0).to(torch.float32).numpy()
    y_val = np.array(val_labels)

    clf = LogisticRegression(random_state=42, max_iter=1000).fit(x_train, y_train)
    
    # test on validation
    y_val_pred = clf.predict(x_val)
    acc = accuracy_score(y_val, y_val_pred)
    
    # zephyr-7b-beta layer 7, LogisticRegression probe on en2fr, hellaswag, acc 99.7
    # zephyr-7b-beta layer 7, LogisticRegression probe on en2es, hellaswag, acc 99.7
    # zephyr-7b-beta layer 7, LogisticRegression probe on en2vi, hellaswag, acc 99.8
    # zephyr-7b-beta layer 7, LogisticRegression probe on en2ja, hellaswag, acc 99.6

    langugae_code = {
        "Vietnamese": "vi",
        "English": "en",
        "French": "fr",
        "Chinese": "zh",
        "Arabic": "ar",
        "Spanish": "es",
        "Russian": "ru",
        "Japanese": "ja",
        "German": "de",
        "Korean": "ko"
    }
    
    save_probes(clf, path=f"data/target_directions/lp/en2{langugae_code[to_language]}/{model_name_or_path}_layer-{layer_id}.pkl")

    print(acc)

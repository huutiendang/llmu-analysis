import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import sys
import pickle
from utils import *
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

def format_truthfulqa(question, choice):
    return f"Q: {question} A: {choice}"

def format_truthfulqa_end_q(question, choice, rand_question): 
    return f"Q: {question} A: {choice} Q: {rand_question}"

def tokenized_tqa_gen(dataset, tokenizer):
    all_prompts = []
    all_labels = []
    all_categories = []
    for i in range(len(dataset)): 
        question = dataset[i]['question']
        category = dataset[i]['category']

        for j in range(len(dataset[i]['correct_answers'])): 
            answer = dataset[i]['correct_answers'][j]
            prompt = format_truthfulqa(question, answer)
            prompt = tokenizer(prompt, return_tensors = 'pt')
            all_prompts.append(prompt)
            all_labels.append(1)
            all_categories.append(category)
        
        for j in range(len(dataset[i]['incorrect_answers'])):
            answer = dataset[i]['incorrect_answers'][j]
            prompt = format_truthfulqa(question, answer)
            prompt = tokenizer(prompt, return_tensors = 'pt')
            all_prompts.append(prompt)
            all_labels.append(0)
            all_categories.append(category)
        
    return all_prompts, all_labels, all_categories

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
    d_train = load_dataset("data/truthfulqa/generation")["train"]
    d_val = load_dataset("data/truthfulqa/generation")["validation"]
    d_test = load_dataset("data/truthfulqa/generation")["test"]

    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1")
    
    train_inputs, train_labels, train_categories = tokenized_tqa_gen(d_train, tokenizer)
    val_inputs, val_labels, val_categories = tokenized_tqa_gen(d_val, tokenizer)
    
    layer_id = 7
    
    hook_module = get_target_module(
        model=model, layer_id=layer_id, module_template="{model_name}.model.layers[{layer_id}]", sub_component=None
    )

    train_activations, val_activations = [], []
    
    for inputs in tqdm.tqdm(train_inputs):
        activations = forward_with_cache(
            model, inputs, module=hook_module, no_grad=False
        ).mean(dim=1).detach().cpu()
        train_activations.append(activations)
    
    x_train = torch.cat(train_activations, dim=0).to(torch.float32).numpy()
    y_train = np.array(train_labels)
    
    # direction_md = find_direction_mean_difference(x_train, y_train)
    
    # save_probes(direction_md, path=f"data/target_directions/md/truth/truth_direction_probe_layer-{layer_id}.pkl")
    
    for inputs in tqdm.tqdm(val_inputs):
        activations = forward_with_cache(
            model, inputs, module=hook_module, no_grad=False
        ).mean(dim=1).detach().cpu()
        val_activations.append(activations)
    
# 
    x_val = torch.cat(val_activations, dim=0).to(torch.float32).numpy()
    y_val = np.array(val_labels)
    # # train cls
    clf = LogisticRegression(random_state=42, max_iter=1000).fit(x_train, y_train)
    # test on validation
    y_val_pred = clf.predict(x_val)
    acc = accuracy_score(y_val, y_val_pred)

    save_probes(clf, path=f"data/target_directions/lp/truth/mistral_truth_direction_probe_layer-{layer_id}.pkl")
    print()

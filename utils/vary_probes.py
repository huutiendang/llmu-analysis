import argparse
from tqdm import tqdm
import torch

from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeClassifier

from transformers import AutoTokenizer, AutoModelForCausalLM

from baselines.utils import save_probes


def load_model(model_name_or_path, tokenizer_name_or_path=None):
    if tokenizer_name_or_path is None:
        tokenizer_name_or_path = model_name_or_path

    torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, 
        trust_remote_code=True, 
        use_fast=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    return model, tokenizer


def get_zscore(data):
    return (data - torch.mean(data, axis=0)) / torch.std(data, axis=0)


def format_truthfulqa(question, choice):
    return f"Q: {question} A: {choice}"


def construct_tqa_gen(dataset):
    all_prompts = []
    all_labels = []
    for i in range(len(dataset)): 
        question = dataset[i]['question']

        for j in range(len(dataset[i]['correct_answers'])): 
            answer = dataset[i]['correct_answers'][j]
            prompt = format_truthfulqa(question, answer)
            all_prompts.append(prompt)
            all_labels.append(1)
        
        for j in range(len(dataset[i]['incorrect_answers'])):
            answer = dataset[i]['incorrect_answers'][j]
            prompt = format_truthfulqa(question, answer)
            all_prompts.append(prompt)
            all_labels.append(0)
        
    return all_prompts, all_labels


def forward_with_cache(model, inputs, module, no_grad=True):
    cache = []

    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None

    hook_handle = module.register_forward_hook(hook)
    with torch.set_grad_enabled(not (no_grad)):
        outputs = model(**inputs)
    hook_handle.remove()
    return cache[0], outputs


def extract_sentence_activation(model, tokenizer, input_text, layer_index):
    module = model.model.layers[layer_index]
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    activations, _ = forward_with_cache(model, inputs, module=module, no_grad=True)
    activations = activations.squeeze(0)

    dist = torch.cdist(activations.float(), activations.float(), p = 1)
    outlier_map = get_zscore(dist.mean(axis=0)) > 3
    activations = activations[~outlier_map]

    return activations.mean(dim=0)


def extract_sentence_activations(model, tokenizer, input_texts, layer_index):
    activations = []
    for input_text in tqdm(input_texts):
        sentence_activation = extract_sentence_activation(model, tokenizer, input_text, layer_index)
        activations.append(sentence_activation.detach())
    return torch.vstack(activations).cpu().to(torch.float32)


def difference_in_means(x_train, y_train):
    x_class_0 = x_train[y_train == 0]
    x_class_1 = x_train[y_train == 1]

    mean_class_0 = torch.mean(x_class_0, dim=0)
    mean_class_1 = torch.mean(x_class_1, dim=0)

    mean_diff = mean_class_1 - mean_class_0
    return mean_diff


def ridge_regression_direction(x_train, y_train):
    clf = RidgeClassifier(random_state=42).fit(x_train, y_train)
    return torch.from_numpy(clf.coef_).squeeze(0)


def kmeans_direction(x_train, y_train, n_clusters=2):
    x_class_0 = x_train[y_train == 0]
    x_class_1 = x_train[y_train == 1]
    mean_class_0 = torch.mean(x_class_0, dim=0)
    mean_class_1 = torch.mean(x_class_1, dim=0)

    kmeans = KMeans(
        n_clusters=n_clusters, 
        init=torch.vstack([mean_class_0, mean_class_1]),
        random_state=42
    ).fit(x_train)
    cluster_centers = torch.from_numpy(kmeans.cluster_centers_).to(torch.float32)

    class_0_cluster_id = torch.cdist(mean_class_0.unsqueeze(0), cluster_centers).argmin().item()
    class_1_cluster_id = torch.cdist(mean_class_1.unsqueeze(0), cluster_centers).argmin().item()

    direction = cluster_centers[class_1_cluster_id] - cluster_centers[class_0_cluster_id]

    return direction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Name of the model")
    parser.add_argument("--layer", type=int, required=True, help="Layer index in the model to evaluate")
    args = parser.parse_args()
    

    model, tokenizer = load_model(args.model)

    d_train = load_dataset("data/truthfulqa/generation")["train"]
    d_val = load_dataset("data/truthfulqa/generation")["validation"]

    train_texts, train_labels = construct_tqa_gen(d_train)
    val_texts, val_labels = construct_tqa_gen(d_val)
    
    X_train = extract_sentence_activations(model, tokenizer, train_texts, args.layer)
    y_train = torch.tensor(train_labels)

    X_val = extract_sentence_activations(model, tokenizer, val_texts, args.layer)
    y_val = torch.tensor(val_labels)

    
    # Difference-in-means
    diff_in_means_dir = difference_in_means(X_train, y_train)
    save_probes(
        diff_in_means_dir, 
        path=f"data/target_directions/difference_in_means/truth/{args.model}_layer-{args.layer}.pkl"
    )

    # K-Means
    kmeans_dir = kmeans_direction(X_train, y_train, n_clusters=2)
    save_probes(
        kmeans_dir, 
        path=f"data/target_directions/k_means/truth/{args.model}_layer-{args.layer}.pkl"
    )

    # Ridge Regression, acc 72.2
    ridge_dir = ridge_regression_direction(X_train, y_train)
    save_probes(
        ridge_dir, 
        path=f"data/target_directions/ridge_regression/truth/{args.model}_layer-{args.layer}.pkl"
    )

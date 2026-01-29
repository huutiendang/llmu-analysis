import os
import datetime
import json
import numpy as np
import torch
# from transformers import AdamW
from torch.optim import AdamW
import tqdm as tqdm
import pickle
# import wandb
# from baselines.utils import load_model, get_params, forward_with_cache, get_data
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_probes(path): 
    with open(path, 'rb') as f: 
        probe = pickle.load(f)
    return probe

def forward_with_cache(model, inputs, module, no_grad=True):
    # define a tensor with the size of our cached activations
    cache = []
    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None 
    
    hook_handle = module.register_forward_hook(hook)
    
    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)
        
    hook_handle.remove()

    return cache[0]
    
#######################################
##### Model and data loading code #####
#######################################


def get_params(model, layer_ids, param_ids):
    params = []
    for layer_id in layer_ids:
        for i, p in enumerate(model.model.layers[layer_id].parameters()):
            if i in param_ids:
                params.append(p)
    return params


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

# def get_random_vector(model, tokenizer, module, keyword="helper", dtype=torch.bfloat16):
#     inputs = tokenizer(keyword, return_tensors="pt", padding=True).to(model.device)
#     activations = forward_with_cache(model, inputs, module)
#     # return activations
#     gaussian_noise = torch.randn([1, 1, activations.shape[-1]])
#     gaussian_noise = gaussian_noise.to(device=model.device, dtype=dtype)
#     gaussian_noise = gaussian_noise / gaussian_noise.norm(dim=-1, keepdim=True)
#     return gaussian_noise
    # if scale is not None:
    #     noise = noise * scale.to(x)
    # return x + noise

# def get_steering_vec(model, tokenizer, keyword, module, token_pos=-1, dtype=torch.bfloat16):

#     p_novice = f"You are a novice in {keyword} who often makes mistakes."
#     p_expert = f"You are a world-class expert in {keyword}."
#     inputs = tokenizer([p_novice, p_expert], return_tensors="pt", padding=True).to(model.device)
#     activations = forward_with_cache(model, inputs, module)

#     # novice - expert
#     direction = activations[0:1, token_pos:, :] - activations[1:, token_pos:, :]
#     # direction = activations.mean(dim=1, keepdim=True)
#     direction = direction.to(device=model.device, dtype=dtype)
#     direction = direction / direction.norm(dim=-1, keepdim=True)
#     return direction

def get_data(forget_corpora, retain_corpora, min_len=0, max_len=2000, batch_size=4):
    def get_dataset(name):
        data = []
        if name == "wikitext":
            from datasets import load_dataset
            raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            for x in raw_data:
                if len(x['text']) > min_len:
                    data.append(str(x['text']))
        else:
            for line in open(f"data/{name}.jsonl", "r"):
                if "bio-forget-corpus" in name:
                    raw_text = json.loads(line)['text']
                else:
                    raw_text = line
                if len(raw_text) > min_len:
                    data.append(str(raw_text))
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        return data
    
    # def corpus_to_keywords(corpus):
    #     if 'bio' in corpus:
    #         return 'bio'
    #     elif 'cyber' in corpus:
    #         return 'cyber'
    #     else:
    #         raise NotImplementedError(f"Add your keywords for {corpus} to the keywords.json file.")

    # with open("data/cyber_kw.json", "r") as f:
    #     keywords = json.load(f)

    return (
        # [keywords['cyber']],
        [get_dataset(c) for c in forget_corpora],
        [get_dataset(c) for c in retain_corpora]
    )

def truncate_models(frozen_model, updated_model, layer_id, layers_str):
    def truncate(model_name):
        layers_var = layers_str.format(model_name=model_name)
        exec(f"{layers_var} = {layers_var}[ : {layer_id + 1}]")
        exec(f"{model_name}.config.num_hidden_layers = {layer_id + 1}")

    truncate("frozen_model")
    truncate("updated_model")


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


def run_rmu(
    updated_model,
    frozen_model,
    tokenizer,
    # keywords_list,
    forget_data_list,
    retain_data_list,
    args,
):
    updated_model = updated_model.train()
    params = get_params(updated_model, args.layer_ids, args.param_ids)
    optimizer = AdamW(params, lr=args.lr)
    
    frozen_module = get_target_module(
        frozen_model, args.layer_id, args.module_str, args.sub_component
    )
    updated_module = get_target_module(
        updated_model, args.layer_id, args.module_str, args.sub_component
    )
    print()
    # steering_vectors_list = [[] for _ in range(len(keywords_list))]
    # for i, keywords in enumerate(keywords_list):
    #     for keyword in keywords:
    #         steering_vectors_list[i].append(
    #             get_steering_vec(
    #                 frozen_model,
    #                 tokenizer,
    #                 keyword,
    #                 frozen_module,
    #             )
    #         )
    # print()
    # control_vectors_list = []
    
    if args.sub_component in ["self_attn.o_proj", "self_attn.k_proj", "self_attn.q_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
        module_shape = updated_module.out_features
    else:
        module_shape = updated_model.config.hidden_size
    
    # module_shape = 
    # for i in range(len(forget_data_list)):
    #     # random_vector = torch.rand(1,1, updated_model.config.hidden_size, dtype=updated_model.dtype, device=updated_model.device)
    #     random_vector = torch.rand(1,1, module_shape, dtype=updated_model.dtype, device=updated_model.device)
    #     control_vec = random_vector / torch.norm(random_vector) * args.steering_coeff_list[i]
    #     control_vectors_list.append(control_vec)
    
    # domain = args.domain
    
    if args.direction in ["positive", "truth", "refusal"]:
        probe = load_probes(path=f"data/target_directions/lp/sentiment/{args.direction}_direction_probe_layer-7.pkl")
        theta = torch.tensor(probe.coef_[0], dtype=torch.bfloat16)
        concept_vector = theta / torch.norm(theta)
        concept_vector = concept_vector.to(updated_model.device)
    
    elif args.direction == "mistral_truth":
        probe = load_probes(path=f"data/target_directions/lp/truth/{args.direction}_direction_probe_layer-7.pkl")
        theta = torch.tensor(probe.coef_[0], dtype=torch.bfloat16)
        concept_vector = theta / torch.norm(theta)
        concept_vector = concept_vector.to(updated_model.device)
        
    elif args.direction == "llama3_refusal":
        probe = load_probes(path=f"data/target_directions/lp/refusal/{args.direction}_direction_probe_layer-7.pkl")
        theta = torch.tensor(probe.coef_[0], dtype=torch.bfloat16)
        concept_vector = theta / torch.norm(theta)
        concept_vector = concept_vector.to(updated_model.device)
    
    elif args.direction == "mistral_negative":
        probe = load_probes(path=f"data/target_directions/lp/sentiment/mistral_positive_direction_probe_layer-7.pkl")
        theta = torch.tensor(probe.coef_[0], dtype=torch.bfloat16)
        concept_vector = - theta / torch.norm(theta)
        concept_vector = concept_vector.to(updated_model.device)
    
    elif args.direction == "mistral_positive":
        probe = load_probes(path=f"data/target_directions/lp/sentiment/mistral_positive_direction_probe_layer-7.pkl")
        theta = torch.tensor(probe.coef_[0], dtype=torch.bfloat16)
        concept_vector = theta / torch.norm(theta)
        concept_vector = concept_vector.to(updated_model.device)
    
    elif args.direction == "negative":
        probe = load_probes(path=f"data/target_directions/lp/sentiment/positive_direction_probe_layer-7.pkl")
        theta = torch.tensor(probe.coef_[0], dtype=torch.bfloat16)
        concept_vector = -theta / torch.norm(theta)
        concept_vector = concept_vector.to(updated_model.device)
    
    elif args.direction == "random":
        rand_vec = torch.randn((1, 1, module_shape), dtype=updated_model.dtype, device=updated_model.device)
        concept_vector = rand_vec / torch.norm(rand_vec)

    num_batches = args.max_num_batches
    
    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side="right"

    # forget_bio, forget_cyber, retain_bio, retain_cyber = {}, {}, {}, {}
    # all_stats = []

    # os.makedirs(f"logs/stats/rm/rmu/{args.model_name_or_path}", exist_ok=True)

    with tqdm.tqdm(total=num_batches) as pbar:
        for idx in range(num_batches):
            topic_idx = idx % len(forget_data_list)
            batch_idx = idx // len(forget_data_list)

            # steering_vecs = steering_vectors_list[topic_idx]
            # steering_vec_idx = np.random.choice(len(steering_vecs))
            # steering_vec = steering_vecs[steering_vec_idx]
            
            unlearn_batch = forget_data_list[topic_idx][batch_idx]
            retain_batch = retain_data_list[topic_idx][batch_idx]

            # Unlearning loss
            max_length = 512 #if topic_idx == 0 else 768

            unlearn_inputs = tokenizer(
                unlearn_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            ).to(updated_model.device)
            updated_forget_activations = forward_with_cache(
                updated_model, unlearn_inputs, module=updated_module, no_grad=False
            ).to(updated_model.device)
            
            frozen_forget_activations = forward_with_cache(
                frozen_model, unlearn_inputs, module=frozen_module, no_grad=True
            ).to(updated_model.device)
            
            
            h_cv = concept_vector.squeeze() 
            
            proj_coeff = torch.einsum('btd,d->bt', frozen_forget_activations, h_cv)
            proj = proj_coeff.unsqueeze(-1) * h_cv
            
            h_t = frozen_forget_activations - args.steering_coeff_list[topic_idx] * proj
            
            unlearn_loss = torch.nn.functional.mse_loss(
                updated_forget_activations, h_t
            )

            # Retain loss
            retain_inputs = tokenizer(
                retain_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            ).to(updated_model.device)
            updated_retain_activations = forward_with_cache(
                updated_model, retain_inputs, module=updated_module, no_grad=False
            ).to(updated_model.device)
            frozen_retain_activations = forward_with_cache(
                frozen_model, retain_inputs, module=frozen_module, no_grad=True
            ).to(updated_model.device)

            # add random noise
            noise = torch.randn_like(frozen_retain_activations) * args.nu
            randomized_frozen_retain_activations = frozen_retain_activations + noise

            retain_loss = torch.nn.functional.mse_loss(
                updated_retain_activations, randomized_frozen_retain_activations
            )
            retain_loss *= args.alpha[topic_idx]

            # Update model
            loss = unlearn_loss + retain_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"loss: {loss.item():.4g} | unlearn_loss: {unlearn_loss.item():.4g} | retain_loss: {retain_loss.item():.4g} | param_change: {params[0].grad.abs().mean().item():.4g}")
        
            # ======= Logging ======
            if args.verbose:
                frozen_forget_activations = forward_with_cache(frozen_model, unlearn_inputs, module=frozen_module, no_grad=True).to(updated_model.device)
                unlearn_cosine= torch.nn.functional.cosine_similarity(updated_forget_activations, frozen_forget_activations, dim=-1).mean()
                retain_cosine = torch.nn.functional.cosine_similarity(updated_retain_activations, frozen_retain_activations, dim=-1).mean()
                
                print(f"unlearn_cosine_sim={unlearn_cosine.item()}")
                print(f"retain_cosine_sim={retain_cosine.item()}")

                updated_forget_norm = torch.mean(
                    updated_forget_activations.norm(dim=-1).mean(dim=1), dim=0
                ).item()
                frozen_forget_norm = torch.mean(
                    frozen_forget_activations.norm(dim=-1).mean(dim=1), dim=0
                ).item()
                updated_retain_norm = torch.mean(
                    updated_retain_activations.norm(dim=-1).mean(dim=1), dim=0
                ).item()
                frozen_retain_norm = torch.mean(
                    frozen_retain_activations.norm(dim=-1).mean(dim=1), dim=0
                ).item()

                print(f"Topic {topic_idx} updated_forget_activations.norm=", updated_forget_norm)
                print(f"Topic {topic_idx} frozen_forget_activations.norm=", frozen_forget_norm)
                print(f"Topic {topic_idx} updated_retain_activations.norm=", updated_retain_norm)
                print(f"Topic {topic_idx} frozen_retain_activations.norm=", frozen_retain_norm)

            pbar.update(1)

    tokenizer.truncation_side = truncation_side
    return updated_model, tokenizer

    # path = f"checkpoints/rm/rmu/{args.model_name_or_path}_alpha-{int(args.alpha[0])}-{int(args.alpha[0])}_coeffs-{float(args.steering_coeff_list[0])}-{float(args.steering_coeff_list[1])}_batches-{num_batches}_layer-{args.layer_id}_nu-{args.nu}"
    # updated_model.save_pretrained(path)
    # tokenizer.save_pretrained(path)
    # print(f"Saved model to {path}")


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    ### Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="HuggingFaceH4/zephyr-7b-beta")
    parser.add_argument("--module_str", type=str, default="{model_name}.model.layers[{layer_id}]")
    parser.add_argument("--sub_component", type=str, default="None", choices=["None", "mlp", "input_layernorm", "self_attn", "self_attn.o_proj", "self_attn.k_proj", "self_attn.q_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "post_attention_layernorm"])
    
    parser.add_argument("--direction", type=str, default="truth", choices=["llama3_refusal", "mistral_positive","mistral_negative", "mistral_truth", "positive", "negative", "random", "truth", "refusal"])
    
    ### Data arguments
    parser.add_argument(
        "--retain_corpora",
        type=str,
        default="wikitext,wikitext",
        help="comma-separated list of corpora to retain",
    )
    parser.add_argument(
        "--forget_corpora",
        type=str,
        default="bio-forget-corpus,cyber-forget-corpus",
        help="comma-separated list of corpora to forget",
    )
    ### rmu hyperparameters
    parser.add_argument("--alpha", type=str, default="1200,1200", help="retain weight")
    parser.add_argument("--steering_coeffs", type=str, default="10,10", help="Steer vector weight in order of topic",)
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--nu", type=float, default=0.00, help="noise magnitude")
    parser.add_argument("--batch_size", type=int, default=4)
    # parser.add_argument('--scale', type=str, default="3.0,3.0")
    parser.add_argument("--max_num_batches", type=int, default=500)
    parser.add_argument("--layer_id", type=int, default=7, help="layer to unlearn")
    parser.add_argument("--layer_ids", type=str, default="5,6,7", help="update layers")
    parser.add_argument("--param_ids", type=str, default="6", help="update params")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--verbose", action="store_true", help="Logging the activations norms and cosine at each step")

    args = parser.parse_args()
    args.retain_corpora = args.retain_corpora.split(",")
    args.forget_corpora = args.forget_corpora.split(",")
    # args.scale = [float(s) for s in args.scale.split(",")]
    args.steering_coeff_list = [float(c) for c in args.steering_coeffs.split(",")]
    args.alpha = [float(c) for c in args.alpha.split(",")]
    args.layer_ids = [int(layer_id) for layer_id in args.layer_ids.split(",")]
    args.param_ids = [int(param_id) for param_id in args.param_ids.split(",")]
    return args 


if __name__ == "__main__":
    args = get_args()

    SEED = args.seed
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    
    frozen_model, tokenizer = load_model(args.model_name_or_path)
    updated_model, _ = load_model(args.model_name_or_path)

    layers_str = args.module_str.split("[")[0]
    truncate_models(frozen_model, updated_model, args.layer_id, layers_str) 

    forget_data_list, retain_data_list = get_data(
        args.forget_corpora,
        args.retain_corpora,
        args.batch_size,
    )

    updated_model, tokenizer = run_rmu(
        updated_model,
        frozen_model,
        tokenizer,
        forget_data_list,
        retain_data_list,
        args,
    )

    path = f"checkpoints/rm/rab/random_direction/{args.model_name_or_path}_alpha-{int(args.alpha[0])}-{int(args.alpha[1])}_coeffs-{float(args.steering_coeff_list[0])}-{float(args.steering_coeff_list[1])}_batches-{args.max_num_batches}_layer-{args.layer_id}_component-{args.sub_component}_nu-{args.nu}"
    
    original_model, _ = load_model(args.model_name_or_path)
    original_layers = eval(layers_str.format(model_name="original_model"))
    updated_layers  = eval(layers_str.format(model_name="updated_model"))
    for l in range(len(updated_layers)):
        original_layers[l] = updated_layers[l]
    
    original_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Saved model to {path}")

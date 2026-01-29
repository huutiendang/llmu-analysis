import json
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
random.seed(0)
import tqdm
import pickle
from torch.autograd import grad
import torch
import torch.nn as nn
################################
##### Activation functions #####
################################

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
        use_fast=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    return model, tokenizer

def get_random_vector(model, tokenizer, module, keyword="helper", dtype=torch.bfloat16):
    inputs = tokenizer(keyword, return_tensors="pt", padding=True).to(model.device)
    activations = forward_with_cache(model, inputs, module)
    # return activations
    gaussian_noise = torch.randn([1, 1, activations.shape[-1]])
    gaussian_noise = gaussian_noise.to(device=model.device, dtype=dtype)
    gaussian_noise = gaussian_noise / gaussian_noise.norm(dim=-1, keepdim=True)
    return gaussian_noise
    # if scale is not None:
    #     noise = noise * scale.to(x)
    # return x + noise

def get_steering_vec(model, tokenizer, keyword, module, dtype=torch.bfloat16):

    # p_novice = f"You are a novice in {keyword} who often makes mistakes."
    # p_expert = f"You are a world-class expert in {keyword}."
    inputs = tokenizer(keyword, return_tensors="pt", padding=True).to(model.device)
    activations = forward_with_cache(model, inputs, module)

    # novice - expert
    # direction = activations[0:1, token_pos:, :] - activations[1:, token_pos:, :]
    direction = activations.mean(dim=1, keepdim=True)
    direction = direction.to(device=model.device, dtype=dtype)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    return direction

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

    return (
        [get_dataset(c) for c in forget_corpora],
        [get_dataset(c) for c in retain_corpora]
    )

def save_probes(probes, path): 
    """takes in a list of sklearn lr probes and saves them to path"""
    with open(path, 'wb') as f: 
        pickle.dump(probes, f)

def load_probes(path): 
    """loads a list of sklearn lr probes from path"""
    with open(path, 'rb') as f: 
        probes = pickle.load(f)
    return probes


# def get_data_idk(forget_corpora, retain_corpora, min_len=0, max_len=2000, batch_size=4):
#     def get_dataset(name):
#         data = []
#         if name == "wikitext":
#             raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
#             for x in raw_data:
#                 if len(x['text']) > min_len:
#                     data.append(str(x['text'][:max_len]))
#         else:
#             for line in open(f"data/{name}.jsonl", "r"):
#                 raw_text = json.loads(line)
#                 if isinstance(raw_text, dict):
#                     raw_text = raw_text.get("text", "")
#                 raw_text = raw_text.strip()
#                 if len(raw_text) > min_len:
#                     data.append(str(raw_text[:max_len]))
#         data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
#         return data 
        
#     # def corpus_to_keywords(corpus):
#     #     if 'bio' in corpus:
#     #         return 'bio'
#     #     elif 'cyber' in corpus:
#     #         return 'cyber'
#     #     elif 'idk' in corpus:
#     #         return 'idk'
#     #     else:
#     #         raise NotImplementedError(f"Add your keywords for {corpus} to the keywords.json file.")

#     with open("data/idontknow.json", "r") as f:
#         keywords = json.load(f)

#     return (
#         [keywords["idk"]],
#         [get_dataset(c) for c in forget_corpora],
#         [get_dataset(c) for c in retain_corpora]
#     )


class RandomizedModel(nn.Module):
    def __init__(self, model_name_or_path: str, target_layers: List[int]):
        super().__init__()
        self.model, self.tokenizer = load_model(model_name_or_path=model_name_or_path)
        self.target_layers = target_layers
    
    def train(self, mode: bool = True):
        super().train(mode)
        self.model.train(mode)
        return self

    def eval(self):
        super().eval()
        self.model.eval()
        return self
    
    def forward_with_injected_noise(self, input_ids, nu, no_grad=True):
        def hook_fn(module, input, output):

            modified_output = list(output)
            temp_hidden_states = modified_output[0]
            noise = torch.randn_like(temp_hidden_states) * nu
            modified_output[0] = temp_hidden_states + noise 
            if isinstance(output, tuple):
                return tuple(modified_output)
            else:
                return modified_output[0]
                
        hooks = []
        
        for layer_ids in self.target_layers:
            if layer_ids < 0 or layer_ids >= len(self.model.model.layers):
                raise ValueError(f"Layer {layer_ids} is invalid.")
            else:
                layer = self.model.model.layers[layer_ids]
                hooks.append(layer.register_forward_hook(hook_fn))
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            
        for hook in hooks:
            hook.remove()
            
        return outputs


    def forward_with_injected_ccv(self, input_ids, vec, nu, no_grad=True):
        def hook_fn(module, input, output):

            modified_output = list(output)
            temp_hidden_states = modified_output[0]
            
            # noise = torch.randn_like(temp_hidden_states) * nu
            concept_vec = vec * nu

            modified_output[0] = temp_hidden_states + concept_vec 
            if isinstance(output, tuple):
                return tuple(modified_output)
            else:
                return modified_output[0]
                
        hooks = []
        
        for layer_ids in self.target_layers:
            if layer_ids < 0 or layer_ids >= len(self.model.model.layers):
                raise ValueError(f"Layer {layer_ids} is invalid.")
            else:
                layer = self.model.model.layers[layer_ids]
                hooks.append(layer.register_forward_hook(hook_fn))
        
        #with torch.no_grad():
        outputs = self.model(input_ids=input_ids, labels=input_ids)
            
        for hook in hooks:
            hook.remove()
            
        return outputs

    def generate(self, prompt, max_length=15, nu=0.01, device="cuda"):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        generated_ids = input_ids.clone()
        
        for _ in range(max_length):
            context_length = min(len(generated_ids[0]), self.model.config.max_position_embeddings)
            input_context = generated_ids[:, -context_length:]
            
            with torch.no_grad():
                outputs = self.forward_with_injected_noise(input_context, nu=nu)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            
            if next_token_id[0, 0].item() == self.tokenizer.eos_token_id:
                break
        generated_text = self.tokenizer.decode(generated_ids[0])
        return generated_text[len(prompt):]

# def exact_hessian(fn, hidden_states):
#     loss = fn(hidden_states)
#     loss_grad = grad(loss, hidden_states, retain_graph=True, create_graph=True)

#     cnt = 0
#     for g in loss_grad:
#         g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
#         cnt = 1
#     l = g_vector.size(0)
#     hessian = torch.zeros(l,l)

#     for idx in tqdm.tqdm(range(l)):
#         grad2rd = grad(g_vector[idx], hidden_states, retain_graph=True, create_graph=False)
#         cnt = 0
#         for g in grad2rd:
#             g2 = g.contiguous(
#             ).view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
#             cnt = 1
#         hessian[idx] = g2
#     return hessian

# def grad_z(loss_fn, hidden_states, create_graph=False):
#     loss = loss_fn(hidden_states)
#     return list(grad(loss, hidden_states, create_graph=create_graph))

# def hvp(y, w, v):
#     """ Multiply the Hessians of y and w by v.
#     Uses a backprop-like approach to compute the product between the Hessian
#     and another vector efficiently, which even works for large Hessians.
#     Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
#     which evaluates to the same values as (A + A.t) v.
#     Arguments:
#         y: scalar/tensor, for example the output of the loss function
#         w: list of torch tensors, tensors over which the Hessian
#             should be constructed
#         v: list of torch tensors, same shape as w,
#             will be multiplied with the Hessian
#     Returns:
#         return_grads: list of torch tensors, contains product of Hessian and v.
#     Raises:
#         ValueError: `y` and `w` have a different length.
#     """
#     if len(w) != len(v):
#         raise(ValueError("w and v must have the same length"))
    
#     first_grads = grad(y, w, retain_graph=True, create_graph=True)

#     # Elementwise products
#     elementwise_products = 0
#     for grad_elem, v_elem in zip(first_grads, v):
#         elementwise_products += torch.sum(grad_elem * v_elem)

#     # second grad
#     return_grads = grad(elementwise_products, w, create_graph=False)

#     return return_grads

# def s_test(fn, hidden_states, num_sample=1, damp=0.01, scale=25.0):
#     v = grad_z(fn, hidden_states, create_graph=False)
#     h_estimate = v.copy() 
#     loss = fn(hidden_states)
#     hv = hvp(loss, hidden_states, h_estimate)
#     h_estimate = [_v + (1 - damp) * _he - _hv/scale for _v, _he, _hv in zip(v, h_estimate, hv)]
#     return h_estimate

# def build_stest(fn, hidden_states, num_iteration=300, scale=25.0):
#     inverse_hvp = [torch.zeros_like(p, dtype=torch.float) for p in hidden_states]
#     cur_estimate = s_test(fn, hidden_states)
#     for r in range(num_iteration):
#         with torch.no_grad():
#             inverse_hvp = [old + (cur/scale) for old, cur in zip(inverse_hvp, cur_estimate)]
#     with torch.no_grad():
#         inverse_hvp = [j / num_iteration for j in inverse_hvp]
#     return inverse_hvp

# from backpack.hessianfree.hvp import hessian_vector_product

# def exact_trace(hidden_states):
#     """Exact trace from sum of Hessian diagonal."""
#     param_trace = hidden_states.diag_h.sum().item()
#     return sum(param_trace)

# def hvp(loss, inputs, v):
#     grads = grad(loss, inputs, create_graph=True)[0]
#     grad_dot_v = torch.sum(grads * v)
#     hv = grad(grad_dot_v, inputs, retain_graph=True)[0]
#     return hv

# def hutchinson_trace_autodiff(fn, hidden_states, epsilon, num_samples=1000):
#     """Hessian trace estimate using autodiff HVPs."""
#     trace = 0.0
#     loss = fn(hidden_states)
#     for _ in tqdm.tqdm(range(num_samples)):
#         Hvec = hessian_vector_product(loss, hidden_states, epsilon)
#         for v, Hv in zip(epsilon, Hvec):
#             vHv = torch.einsum("i,i->", v.flatten(), Hv.flatten())
#             trace += vHv / num_samples
#     return trace / num_samples

if __name__ == "__main__":
    # model_name = "HuggingFaceH4/zephyr-7b-beta"
    # model_name = "meta-llama/Meta-Llama-3-8B"
    # model_name = "01-ai/Yi-6B"
    # model_name = "microsoft/Phi-3.5-mini-instruct"
    # model_name = "allenai/OLMoE-1B-7B-0924"
    # model_name = "tiiuae/falcon-mamba-7b"
    # model, _ = load_model(model_name_or_path=model_name)
    # randomized_model = RandomizedModel(model_name_or_path=model_name, target_layers=[7], eta=0.001)
    
    # text = "What is the capital of France?"
    
    # # output = model()
    # inputs = randomized_model.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
    # input_ids = inputs.input_ids.to("cuda")
    # # attention_mask = inputs.attention_mask.to("cuda")
    # output = model(**inputs)
    # logits = randomized_model.forward_with_injected_noise(input_ids=input_ids)
    print()

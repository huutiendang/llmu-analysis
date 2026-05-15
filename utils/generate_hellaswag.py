from argparse import ArgumentParser

import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams


PROMPT_TEMPLATE = "Finish this sentence: {ctx} Answer:"


def build_prompts(ds, language=None):
    prompts = []

    for example in ds:
        ctx = example["ctx"]

        prompt = PROMPT_TEMPLATE.format(ctx=ctx)
        if language is not None:
            prompt += f" in {language}:"
        prompts.append(prompt)

    return prompts


def generate_qa(ds, language, llm, sampling_params):
    prompts = build_prompts(ds, language)

    outputs = llm.generate(prompts, sampling_params)
    results = [out.outputs[0].text.strip() for out in outputs]

    ds = ds.add_column("prompt", prompts)
    ds = ds.add_column("qa_generation", results)
    return ds


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    llm = LLM(model=args.model_name_or_path, dtype=torch.bfloat16, max_model_len=4096)
    sampling_params = SamplingParams(max_tokens=100, 
                                     temperature=0.0, 
                                     top_k=1,
                                     stop=["</s>", "<|im_end|>"],
                                     seed=1234)

    ds = load_dataset("Rowan/hellaswag", split="validation")
    ds = generate_qa(ds, args.language, llm, sampling_params)

    ds.to_json(
        f"{args.output_dir}/hellaswag.jsonl" if args.language is None else f"{args.output_dir}/hellaswag_in_{args.language}.jsonl",
        orient="records",
        lines=True,
        force_ascii=False
    )

import json
from argparse import ArgumentParser
from pathlib import Path

import datasets
import evaluate
import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from trl import SFTTrainer

parse = ArgumentParser()
parse.add_argument("model_name")
parse.add_argument("save_path")
parse.add_argument("--data_path", default="./data/")

LABELS_VOCAB = ["O", "PRED", "OBJ", "ASP"]
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
SYSTEM_PROMPT = (
    B_SYS
    + """You are a helpfull assistnat for sequence labeling with the following labels: OBJ - Object, ASP - Aspect, PRED - Predicate and O - none.

For example:
Input: [ID-0]who [ID-1]was [ID-2]the [ID-3]longest [ID-4]ruling [ID-5]non [ID-6]royal [ID-7]head [ID-8]of [ID-9]state [ID-10]post [ID-11]1900
Output: [ID-0][O] [ID-1][O] [ID-2][O] [ID-3][PRED] [ID-4][ASP] [ID-5][OBJ] [ID-6][OBJ] [ID-7][OBJ] [ID-8][OBJ] [ID-9][OBJ] [ID-10][OBJ] [ID-11][OBJ]

Input: [ID-0]what [ID-1]are [ID-2]the [ID-3]differences [ID-4]between [ID-5]through [ID-6]and [ID-7]thru
Output: [ID-0][O] [ID-1][O] [ID-2][O] [ID-3][PRED] [ID-4][O] [ID-5][O] [ID-6][O] [ID-7][OBJ]

Input: [ID-0]which [ID-1]was [ID-2]the [ID-3]first [ID-4]national [ID-5]park
Output: [ID-0][O] [ID-1][O] [ID-2][O] [ID-3][PRED] [ID-4][OBJ] [ID-5][OBJ]
"""
    + E_SYS
)

LABELS_VOCAB = ["O", "PRED", "OBJ", "ASP"]


def gen_prompt(input_text, instruction):
    return B_INST + SYSTEM_PROMPT + instruction + input_text + E_INST


def group_to_examples(group):
    input_sentence = " ".join(
        [f"[ID-{id}]{token}" for id, token in enumerate(group["words"].values)]
    )
    target_sentence = " ".join(
        [f"[ID-{id}][{label}]" for id, label in enumerate(group["labels"].values)]
    )
    return input_sentence, target_sentence


def df_to_dataset(df):
    dataset = []
    for group_id, group in tqdm(
        df.groupby("sentence_id"), total=len(df.sentence_id.unique())
    ):
        input_sentence, target_sentence = group_to_examples(group)
        inputs = gen_prompt("Input: " + input_sentence + "\n", "")
        dataset.append(
            {
                "input_text": inputs,
                "output_text": "Output: " + target_sentence,
                "text": inputs + "Output: " + target_sentence + "</s>",
            }
        )
    return datasets.Dataset.from_pandas(pd.DataFrame(dataset))


if __name__ == "__main__":
    args = parse.parse_args()

    ds = datasets.DatasetDict(
        {
            "train": df_to_dataset(pd.read_csv(Path(args.data_path) / "train.csv")),
            "validation": df_to_dataset(
                pd.read_csv(Path(args.data_path) / "valid.csv")
            ),
            "test": df_to_dataset(pd.read_csv(Path(args.data_path) / "test.csv")),
        }
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        quantization_config=bnb_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    model.config.pad_token_id = tokenizer.pad_token_id

    pipe = pipeline(
        task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128
    )

    final_outputs = []
    final_targets = []
    for input in tqdm(DataLoader(ds["test"], batch_size=32)):
        output_text = pipe(input["input_text"])

        generated_text = [output[0]["generated_text"] for output in output_text]
        generated_seq = [t.split("Output: ")[-1].split(" ") for t in generated_text]

        target_text = input["text"]
        target_seqs = []
        for seq in [t.split("Output: ")[-1].split(" ") for t in target_text]:
            new_seq = []
            for token in seq:
                new_token = token.split("[")[-1].split("]")[0]
                if new_token in LABELS_VOCAB:
                    new_seq.append("T-" + new_token)
                else:
                    new_seq.append("T-O")
            target_seqs.append(new_seq)

        gen_seqs = []
        for seq_idx, seq in enumerate(target_seqs):
            gen_seq = ["T-O"] * len(seq)
            for token in generated_seq[seq_idx]:
                try:
                    id = int(token.split("]")[0].split("-")[-1])
                    gen_seq[id] = token.split("[")[-1].split("]")[0]
                    if gen_seq[id] in LABELS_VOCAB:
                        gen_seq[id] = "T-" + gen_seq[id]
                    else:
                        gen_seq[id] = "T-O"
                except:
                    pass
            gen_seqs.append(gen_seq)

        final_targets.extend(target_seqs)
        final_outputs.extend(gen_seqs)

    metric = evaluate.load("seqeval", scheme="IOB2")

    results = metric.compute(
        predictions=final_outputs,
        references=final_targets,
    )

    results_unfolded = {}
    for key, value in results.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                results_unfolded[str(key + "_" + subkey)] = float(subvalue)
        else:
            results_unfolded[str(key)] = float(value)

    for key, val in results_unfolded.items():
        print(key, val)

    with open(args.save_path, "w") as f:
        json.dump(results_unfolded, f)

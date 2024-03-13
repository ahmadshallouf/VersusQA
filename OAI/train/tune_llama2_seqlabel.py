from pathlib import Path

import datasets
import evaluate
import numpy as np
import pandas as pd
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
)
from trl import SFTTrainer

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


metric = evaluate.load("seqeval", scheme="IOB2")


def compute_metrics(eval_preds):
    preds = eval_preds.predictions
    label_ids = eval_preds.label_ids

    preds = np.argmax(preds, axis=2)
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    generated_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
    generated_seq = [t.split("Output: ")[-1].split(" ") for t in generated_text]

    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
    target_text = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
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

    # raise Exception(str([
    #     f"{pred}\n{targ}\n\n"
    #     for pred, targ in zip(gen_seqs, target_seqs)
    # ]))
    results = metric.compute(
        predictions=gen_seqs,
        references=target_seqs,
    )

    results_unfolded = {}
    for key, value in results.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                results_unfolded[key + "_" + subkey] = subvalue
        else:
            results_unfolded[key] = value

    return results_unfolded


if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b-hf"
    # model_name = "lmsys/vicuna-7b-v1.5"
    output_dir = "./tmp_llama_seqlabeling"
    num_train_epochs = 8
    max_seq_length = 856
    per_device_train_batch_size = 2
    gradient_accumulation_steps = 16
    per_device_eval_batch_size = 2
    eval_accumulation_steps = 64
    gradient_checkpointing = True
    max_grad_norm = 0.3
    learning_rate = 2e-4
    weight_decay = 0.001
    optim = "paged_adamw_32bit"
    lr_scheduler_type = "cosine"
    max_steps = -1
    warmup_ratio = 0.03
    group_by_length = True
    save_steps = 0
    logging_steps = 50
    eval_steps = 100
    bf16 = True

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    ds = datasets.DatasetDict(
        {
            "train": df_to_dataset(pd.read_csv("./data/train.csv")),
            "validation": df_to_dataset(pd.read_csv("./data/valid.csv")),
            "test": df_to_dataset(pd.read_csv("./data/test.csv")),
        }
    )

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_accumulation_steps=eval_accumulation_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=False,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="wandb",
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"].select(range(32)),
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
        compute_metrics=compute_metrics,
    )

    # trainer.evaluate()
    trainer.train()
    trainer.model = trainer.model.merge_and_unload()
    trainer.model.save_pretrained(Path(output_dir) / "fitted_model" / model_name)

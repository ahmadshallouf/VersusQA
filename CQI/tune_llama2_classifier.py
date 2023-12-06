import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    AutoModelForSequenceClassification,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, PeftModel, get_peft_model
import pandas as pd
import evaluate
import numpy as np
from argparse import ArgumentParser
from pathlib import Path

parse = ArgumentParser()
parse.add_argument("model_name")
parse.add_argument("output_dir")
parse.add_argument("--seed", default=42, type=int)
parse.add_argument("--dataset", choices=['comparg', 'comp_ident'], default='comp_ident')


def load_comparg_data():
    train_df = pd.read_csv(
    './data/comparg_train.tsv',
    on_bad_lines='warn',
    sep='\t',
    )
    train_df['labels'] = train_df['labels'].apply(
        lambda x: x - 1 if x > 0 else 0
    )
    
    test_df = pd.read_csv(
        './data/comparg_test.tsv',
        on_bad_lines='warn',
        sep='\t',
    )
    test_df['labels'] = test_df['labels'].apply(
        lambda x: x - 1 if x > 0 else 0
    )
    return train_df, test_df


def load_comp_ident_data():
    train_df = pd.read_csv(
        './train_paper.csv',
        on_bad_lines='warn',
        sep='\t',
    )
    
    valid_df = pd.read_csv(
        './validate_paper.csv',
        on_bad_lines='warn',
        sep='\t',
    )
    
    test_df = pd.read_csv(
        './test_paper.csv',
        on_bad_lines='warn',
        sep='\t',
    )
    return train_df, test_df


def get_preprocessed_comparg_dataset(
    dataset,
    tokenizer,
    max_seq_length,
    padding,
    preprocessing_num_workers=8,
    ignore_pad_token_for_loss=True,
):
    def preprocess_function(examples):
        inputs = []
        for obj0, obj1, ans in zip(examples['object_0'], examples['object_1'], examples['answer']):
            inputs.append(
                "Object 1: " + str(obj0) + "; " +
                "Object 2: " + str(obj1) + "; " + 
                ans
            )

        model_inputs = tokenizer(
            inputs, max_length=max_seq_length, padding=padding, truncation=True
        )
        labels = torch.tensor(examples['labels'], dtype=torch.long)
        model_inputs["labels"] = labels
        return model_inputs

    column_names = dataset.column_names
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=column_names,
        # load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    return dataset


def get_preprocessed_comp_ident_dataset(
    dataset,
    tokenizer,
    max_seq_length,
    padding,
    preprocessing_num_workers=8,
    ignore_pad_token_for_loss=True,
):
    def preprocess_function(examples):
        inputs = examples['question']

        model_inputs = tokenizer(
            inputs, max_length=max_seq_length, padding=padding, truncation=True
        )
        labels = torch.tensor(examples['category'], dtype=torch.long)
        model_inputs["labels"] = labels
        return model_inputs

    column_names = dataset.column_names
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=column_names,
        # load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    return dataset


def load_all(model_name, output_dir, dataset, seed=42):
    ################################################################################
    # QLoRA parameters
    ################################################################################
    
    # LoRA attention dimension
    lora_r = 64
    
    # Alpha parameter for LoRA scaling
    lora_alpha = 16
    
    # Dropout probability for LoRA layers
    lora_dropout = 0.1
    
    ################################################################################
    # bitsandbytes parameters
    ################################################################################
    
    # Activate 4-bit precision base model loading
    use_4bit = False
    
    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"
    
    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "fp4"
    
    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = False
    
    ################################################################################
    # TrainingArguments parameters
    ################################################################################
    
    # Enable fp16/bf16 training (set bf16 to True with an A100)
    fp16 = False
    bf16 = True
    
    # Batch size per GPU for training
    per_device_train_batch_size = 2
    
    # Batch size per GPU for evaluation
    per_device_eval_batch_size = 2
    
    # Number of update steps to accumulate the gradients for
    gradient_accumulation_steps = 32
    
    # Enable gradient checkpointing
    gradient_checkpointing = True
    
    # Maximum gradient normal (gradient clipping)
    max_grad_norm = 0.3
    
    # Initial learning rate (AdamW optimizer)
    learning_rate = 1e-4
    
    # Weight decay to apply to all layers except bias/LayerNorm weights
    weight_decay = 1e-3
    
    # Optimizer to use
    optim = "paged_adamw_32bit"
    lr_scheduler_type = "cosine"
    max_steps = -1
    warmup_ratio = 0.03
    group_by_length = True
    save_steps = 0
    logging_steps = 100
    eval_steps = 100
    eval_accumulation_steps = 16
    num_train_epochs = 15

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    
    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)
    
    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        # model_name,
        output_dir,
        quantization_config=bnb_config,
        device_map="auto",
        num_labels=3,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # # Load LoRA configuration
    # peft_config = LoraConfig(
    #     lora_alpha=lora_alpha,
    #     lora_dropout=lora_dropout,
    #     r=lora_r,
    #     bias="none",
    #     task_type="SEQ_CLS",
    # )
    
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()


    # PREPARE DATA
    if dataset == 'comp_ident':
        train_df, test_df = load_comp_ident_data()
        get_preprocessed_dataset = get_preprocessed_comp_ident_dataset
    else:
        train_df, test_df = load_comparg_data()
        get_preprocessed_dataset = get_preprocessed_comparg_dataset
    
    train_ds = Dataset.from_pandas(train_df)
    train_ds = get_preprocessed_dataset(train_ds, tokenizer, 768, 'max_length')
    
    test_ds = Dataset.from_pandas(test_df)
    test_ds = get_preprocessed_dataset(test_ds, tokenizer, 768, 'max_length')
    columns = [
        "input_ids",
        "labels",
        "attention_mask",
    ]
    train_ds.set_format(type="torch", columns=columns)
    test_ds.set_format(type="torch", columns=columns)
    
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        results = f1.compute(predictions=predictions, references=labels, average='weighted')
        results.update(precision.compute(predictions=predictions, references=labels, average='weighted'))
        results.update(recall.compute(predictions=predictions, references=labels, average='weighted'))
        return results

    training_arguments = TrainingArguments(
        report_to='wandb',
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        eval_accumulation_steps=eval_accumulation_steps,
        eval_steps=eval_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        # no_cuda=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    early_stopping = EarlyStoppingCallback(early_stopping_patience=2)
    trainer.add_callback(early_stopping)

    return trainer, (train_ds, test_ds), output_dir


def fit_and_save(trainer, train_ds, test_ds, output_dir, model_name):
    trainer.train()

    trainer.evaluate(test_ds)

    trainer.model = trainer.model.merge_and_unload()
    trainer.model.save_pretrained(Path(output_dir) / 'fitted_model' / model_name)


if __name__ == '__main__':
    args = parse.parse_args()
    print(args)
    
    trainer, (train_ds, test_ds), output_dir = load_all(
        args.model_name,
        args.output_dir,
        args.dataset,
        args.seed,
    )
    fit_and_save(trainer, train_ds, test_ds, output_dir, args.model_name)
    
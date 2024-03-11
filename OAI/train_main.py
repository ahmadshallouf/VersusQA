import os
import random

import evaluate
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset, DatasetDict
from ray import tune
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

CONFIG_PATH = ""
LABEL_LIST = ["O", "B-OBJ", "I-OBJ", "B-ASP", "I-ASP", "B-PRED", "I-PRED"]


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


config = load_config("configuration.yaml")

eval_inputs = None
tokenizer = None
tokenized_datasets = None


def tokenize_and_align_labels(examples, label_all_tokens=False, **kwargs):
    # deberta first word starts with ▁
    # next word with ' ' nothing
    tokenizer = kwargs["tokenizer"]

    tokenized_inputs = tokenizer(
        examples["words"], truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        label_ids = [
            LABEL_LIST.index(idx) if isinstance(idx, str) else idx for idx in label_ids
        ]

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(eval_preds):
    predictions, labels, inputs = (
        eval_preds.predictions,
        eval_preds.label_ids,
        eval_inputs,
    )
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []
    for prediction, label, tokens in zip(predictions, labels, inputs):
        true_predictions.append([])
        true_labels.append([])
        for p, l, t in zip(prediction, label, tokens):
            if l != -100 and tokenizer.convert_ids_to_tokens(int(t)).startswith("__"):
                true_predictions[-1].append(LABEL_LIST[p])
                true_labels[-1].append(LABEL_LIST[l])

    metric = evaluate.load("seqeval", scheme="IOB2")
    results = metric.compute(predictions=true_predictions, references=true_labels)
    results_unfolded = {}
    for key, value in results.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                results_unfolded[key + "_" + subkey] = subvalue
        else:
            results_unfolded[key] = value
    return results_unfolded


def raytune_hp_space(trial):
    return {
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "per_device_train_batch_size": tune.choice([8, 16]),
        "num_train_epochs": tune.choice([3, 4, 5, 6]),
        # "weight_decay": tune.loguniform(1e-3, 1e-1),
        # "warmup_steps": tune.uniform(0, 500),
        # "seed": tune.uniform(2, 42),
    }


def model_init():
    model = AutoModelForTokenClassification.from_pretrained(
        config["model"]["name"],
        num_labels=len(LABEL_LIST),
        ignore_mismatched_sizes=True,
    ).to(config["device"])
    model.config.id2label = dict(enumerate(LABEL_LIST))
    model.config.label2id = {v: k for k, v in model.config.id2label.items()}
    return model


def transform_to_iob2_format(labels):
    new_labels = []
    prev_label = labels[0]
    is_first_label = True
    for ind in range(1, len(labels)):
        label = labels[ind]
        if prev_label != label:
            new_label = "B-" + prev_label if is_first_label else "I-" + prev_label
            new_labels.append(prev_label if prev_label == "O" else new_label)
            prev_label = label
            is_first_label = True
        elif is_first_label:
            new_labels.append(prev_label if prev_label == "O" else "B-" + prev_label)
            prev_label = label
            is_first_label = False
        else:
            new_labels.append(prev_label if prev_label == "O" else "I-" + prev_label)
            prev_label = label

    new_label = "B-" + prev_label if is_first_label else "I-" + prev_label
    new_labels.append(prev_label if prev_label == "O" else new_label)

    return new_labels


def read_data(filename):
    df = (
        pd.read_csv(config["data"]["folder_path"] + filename, sep=",")
        .groupby("sentence_id")
        .agg({"words": lambda x: list(x), "labels": lambda x: list(x)})
    )
    df = df.reset_index(drop=True)
    df["labels"] = df["labels"].map(lambda x: transform_to_iob2_format(x))
    return df


def evaluate_dataset(trainer, dataset_part_name):
    global tokenized_datasets
    global eval_inputs
    global tokenizer

    eval_inputs = tokenized_datasets[dataset_part_name]["input_ids"]
    results = trainer.predict(
        test_dataset=tokenized_datasets[dataset_part_name],
        metric_key_prefix=dataset_part_name,
    )
    inputs = eval_inputs

    predictions = np.argmax(results.predictions, axis=2)

    true_predictions = []
    true_labels = []
    for prediction, label, tokens in zip(predictions, results.label_ids, inputs):
        true_predictions.append([])
        true_labels.append([])
        for p, l, t in zip(prediction, label, tokens):
            if l != -100 and tokenizer.convert_ids_to_tokens(int(t)).startswith("▁"):
                true_predictions[-1].append(LABEL_LIST[p])
                true_labels[-1].append(LABEL_LIST[l])

    df = pd.DataFrame(
        {
            "words": tokenizer.batch_decode(inputs),
            "labels": true_labels,
            "predictions": true_predictions,
        }
    )
    df.to_csv(f"{config['log']['run_name']}-{dataset_part_name}.csv", index=False)


def train_main():
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])

    os.environ["WANDB_PROJECT"] = "draft-" + config["log"]["run_name"]
    os.environ["WANDB_LOG_MODEL"] = "end"
    os.environ["WANDB_WATCH"] = "all"
    os.environ["WANDB_SILENT"] = "false"

    """# Data loading"""

    train = read_data("train.csv")
    val = read_data("valid.csv")
    test = read_data("test.csv")

    ner_data = DatasetDict(
        {
            "train": Dataset.from_pandas(train),
            "valid": Dataset.from_pandas(val),
            "test": Dataset.from_pandas(test),
        }
    )

    """# Training
    """

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"],
        model_max_length=config["model"]["model_max_length"],
        add_prefix_space=True,
    )

    global tokenized_datasets
    tokenized_datasets = ner_data.map(
        tokenize_and_align_labels,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    args = TrainingArguments(
        output_dir=f"./{config['log']['run_name']}-finetuned-obj/",
        overwrite_output_dir=f"./{config['log']['run_name']}-finetuned-obj/",
        seed=config["seed"],
        data_seed=config["seed"],
        run_name=config["log"]["run_name"],
        load_best_model_at_end=f"./{config['log']['run_name']}-best-end/",
        metric_for_best_model=config["model"]["metric_for_best"],
        evaluation_strategy=config["eval"]["strategy"],
    )
    args.set_dataloader(
        sampler_seed=config["seed"],
        train_batch_size=config["train"]["batch_size"],
        eval_batch_size=config["eval"]["batch_size"],
    )  # auto_find_batch_size=True
    args.set_evaluate(
        strategy=config["eval"]["strategy"],
        steps=config["eval"]["steps"],
        delay=config["eval"]["delay"],
        batch_size=config["eval"]["batch_size"],
    )
    args.set_logging(
        strategy=config["log"]["strategy"],
        steps=config["log"]["steps"],
        report_to=config["log"]["report_to"],
        first_step=config["log"]["first_step"],
        level=config["log"]["level"],
    )
    # args.set_lr_scheduler(name=config["lr_name"],
    #                       warmup_steps=config["lr_warmup_steps"])

    args.set_optimizer(
        name=config["optimizer_name"],
        learning_rate=config["optimizer_learning_rate"],
    )
    #                  weight_decay=config["lr_weight_decay"])
    # args.set_save(strategy=config["save"]["strategy"], steps=config["save"]["steps"])
    args.set_testing(batch_size=config["test"]["batch_size"])
    args.set_training(
        num_epochs=config["train"]["num_epochs"],
        batch_size=config["train"]["batch_size"],
    )

    global eval_inputs
    eval_inputs = tokenized_datasets["valid"]["input_ids"]

    trainer = Trainer(
        model=model_init(),
        # model_init=model_init,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # print("Hyperparameter Search")
    # best_trial = trainer.hyperparameter_search(
    #     direction="maximize",
    #     backend="ray",
    #     hp_space=raytune_hp_space,
    #     n_trials=10,
    #     scheduler=PopulationBasedTraining(metric="objective", mode="max",
    #                                       hyperparam_mutations=raytune_hp_space("trial")),
    # )

    # print("Best trial:", best_trial)

    print("Training")
    trainer.train()

    print("Inference")
    results_file = open(f"{config['log']['run_name']}-final.txt", "w")
    results_file.write("Training\n")
    eval_inputs = tokenized_datasets["train"]["input_ids"]
    results = trainer.evaluate(eval_dataset=tokenized_datasets["train"])
    results_file.writelines([f"{results}", "\n"])

    results_file.write("Validating\n")
    eval_inputs = tokenized_datasets["valid"]["input_ids"]
    results = trainer.evaluate(eval_dataset=tokenized_datasets["valid"])
    results_file.writelines([f"{results}", "\n"])

    results_file.write("Testing\n")
    eval_inputs = tokenized_datasets["test"]["input_ids"]
    results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    results_file.writelines([f"{results}", "\n"])
    results_file.close()

    print("Save the model")
    trainer.save_model(f"./{config['log']['run_name']}-best/")


if __name__ == "__main__":
    train_main()

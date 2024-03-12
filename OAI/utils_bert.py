import os

import evaluate
import numpy as np
import pandas as pd
import yaml
from transformers import AutoModelForTokenClassification

CONFIG_PATH = ""
LABEL_LIST = ["O", "B-OBJ", "I-OBJ", "B-ASP", "I-ASP", "B-PRED", "I-PRED"]


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


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


def read_data(filename, config=load_config("configuration.yaml")):
    df = (
        pd.read_csv(config["data"]["folder_path"] + filename, sep=",")
        .groupby("sentence_id")
        .agg({"words": lambda x: list(x), "labels": lambda x: list(x)})
    )
    df = df.reset_index(drop=True)
    df["labels"] = df["labels"].map(lambda x: transform_to_iob2_format(x))
    return df


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


def model_init_helper(config=load_config("configuration.yaml")):
    def model_init():
        nonlocal config
        print(config)
        print("djhsdglh js")
        model = AutoModelForTokenClassification.from_pretrained(
            config["model"]["name"],
            num_labels=len(LABEL_LIST),
            ignore_mismatched_sizes=True,
        )
        model.config.id2label = dict(enumerate(LABEL_LIST))
        model.config.label2id = {v: k for k, v in model.config.id2label.items()}
        return model

    return model_init


def compute_metrics_helper(tokenizer):
    def compute_metrics(eval_preds):
        nonlocal tokenizer
        predictions, labels, inputs = (
            eval_preds.predictions,
            eval_preds.label_ids,
            eval_preds.inputs,
        )
        predictions = np.argmax(predictions, axis=2)

        true_predictions = []
        true_labels = []
        for prediction, label, tokens in zip(predictions, labels, inputs):
            true_predictions.append([])
            true_labels.append([])
            for p, l, t in zip(prediction, label, tokens):
                if l != -100 and tokenizer.convert_ids_to_tokens(int(t)).startswith(
                    "▁"
                ):
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

    return compute_metrics


def evaluate_dataset(
    trainer, tokenized_dataset, metric_prefix, config=load_config("configuration.yaml")
):
    results = trainer.predict(
        test_dataset=tokenized_dataset,
        metric_key_prefix=metric_prefix,
    )

    predictions = np.argmax(results.predictions, axis=2)

    true_predictions = []
    true_labels = []
    for prediction, label, tokens in zip(
        predictions, results.label_ids, results.inputs
    ):
        true_predictions.append([])
        true_labels.append([])
        for p, l, t in zip(prediction, label, tokens):
            if l != -100 and trainer.tokenizer.convert_ids_to_tokens(int(t)).startswith(
                "▁"
            ):
                true_predictions[-1].append(LABEL_LIST[p])
                true_labels[-1].append(LABEL_LIST[l])

    df = pd.DataFrame(
        {
            "words": trainer.tokenizer.batch_decode(results.inputs),
            "labels": true_labels,
            "predictions": true_predictions,
        }
    )
    df.to_csv(f"{config['log']['run_name']}-{metric_prefix}.csv", index=False)

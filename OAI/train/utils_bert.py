import copy
import os

import evaluate
import numpy as np
import pandas as pd
import yaml
from transformers import AutoModelForTokenClassification

CONFIG_PATH = "/home/user/VersusQA/OAI/train/"
LABEL_LIST = ["O", "OBJ", "ASP", "PRED"]


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


def read_data(filename, config=load_config("config.yaml")):
    df = (
        pd.read_csv(config["data"]["folder_path"] + filename, sep=",")
        .groupby("sentence_id")
        .agg({"words": lambda x: list(x), "labels": lambda x: list(x)})
    )
    df["labels"] = df["labels"].apply(
        lambda labels: [LABEL_LIST.index(label) for label in labels]
    )
    df = df.reset_index(drop=True)
    # df["labels"] = df["labels"].map(lambda x: transform_to_iob2_format(x))
    return df


def tokenize_and_align_labels(examples, one_label_per_word=True, **kwargs):
    tokenizer = kwargs["tokenizer"]

    tokenized_inputs = tokenizer(
        examples["words"], truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        current_word = None
        new_labels = []
        for word_id in word_ids:
            if word_id is None:
                new_labels.append(-100)
            elif word_id != current_word:
                current_word = word_id
                new_labels.append(label[word_id])
            else:
                new_labels.append(-100 if one_label_per_word else label[word_id])
        labels.append(new_labels)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def model_init_helper(config=load_config("config.yaml")):
    def model_init():
        nonlocal config
        id2label = {i: label for i, label in enumerate(LABEL_LIST)}
        label2id = {v: k for k, v in id2label.items()}
        model = AutoModelForTokenClassification.from_pretrained(
            config["model"]["name"],
            id2label=id2label,
            label2id=label2id,
        )
        print(f"{model.config.num_labels = }")
        return model

    return model_init


def compute_metrics(eval_preds):
    predictions, labels = (
        eval_preds.predictions,
        eval_preds.label_ids,
    )
    predictions = np.argmax(predictions, axis=-1)

    true_labels = [[LABEL_LIST[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    metric = evaluate.load("seqeval")
    results = metric.compute(predictions=true_predictions, references=true_labels)
    results_unfolded = {}
    for key, value in results.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                results_unfolded[key + "_" + subkey] = subvalue
        else:
            results_unfolded[key] = value
    return results_unfolded


def compute_objective(metrics):
    metrics = copy.deepcopy(metrics)
    return metrics["eval_overall_f1"]


def evaluate_dataset(
    trainer, tokenized_dataset, metric_prefix, config=load_config("config.yaml")
):
    results = trainer.predict(
        test_dataset=tokenized_dataset,
        metric_key_prefix=metric_prefix,
    )

    predictions, labels = (
        results.predictions,
        results.label_ids,
    )
    predictions = np.argmax(predictions, axis=-1)

    true_labels = [[LABEL_LIST[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    df = pd.DataFrame(
        {
            "words": trainer.tokenizer.batch_decode(results.inputs),
            "labels": true_labels,
            "predictions": true_predictions,
        }
    )
    df.to_csv(f"{config['log']['run_name']}-{metric_prefix}.csv", index=False)


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

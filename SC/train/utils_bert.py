import os

import evaluate
import numpy as np
import yaml
from transformers import AutoModelForSequenceClassification

CONFIG_PATH = "./"


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


def tokenize_function(examples, **kwargs):
    tokenizer = kwargs["tokenizer"]
    return tokenizer(examples["text"])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    f1_metric = evaluate.load("f1")
    recall_metric = evaluate.load("recall")
    precision_metric = evaluate.load("precision")
    accuracy_metric = evaluate.load("accuracy")

    results = {}
    results.update(
        f1_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )
    )
    results.update(
        recall_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )
    )
    results.update(
        precision_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )
    )
    results.update(accuracy_metric.compute(predictions=predictions, references=labels))

    return results


def model_init(config=load_config("config.yaml")):
    return AutoModelForSequenceClassification.from_pretrained(
        config["model"]["name"], num_labels=4
    )

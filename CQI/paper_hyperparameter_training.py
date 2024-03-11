import gc

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from datasets.dataset_dict import DatasetDict
from ray import tune
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

"""
Models use in trials:

distilbert-base-uncased-finetuned-sst-2-english
distilbert-base-uncased
prajjwal1/bert-tiny
roberta-base
microsoft/deberta-base
"""


def hp_space(trial):
    return {
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "per_device_train_batch_size": tune.choice([8, 12, 16]),
        "num_train_epochs": tune.choice([3, 4, 5, 6]),
        "seed": tune.uniform(2, 42),
    }


torch.cuda.empty_cache()
gc.collect()

train = pd.read_csv("train_paper.csv", header=0)
val = pd.read_csv("validate_paper.csv", header=0)
test = pd.read_csv("test_paper.csv", header=0)

d = {
    "train": Dataset.from_dict(
        {
            "text": train["question"].values.tolist(),
            "label": train["comp"].values.tolist(),
        }
    ),
    "test": Dataset.from_dict(
        {
            "text": test["question"].values.tolist(),
            "label": test["comp"].values.tolist(),
        }
    ),
    "validation": Dataset.from_dict(
        {"text": val["question"].values.tolist(), "label": val["comp"].values.tolist()}
    ),
}

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english", model_max_length=max_length
)

f1_metric = evaluate.load("f1")
recall_metric = evaluate.load("recall")
precision_metric = evaluate.load("precision")
accuracy_metric = evaluate.load("accuracy")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    results = {}
    results.update(f1_metric.compute(predictions=predictions, references=labels))
    results.update(recall_metric.compute(predictions=predictions, references=labels))
    results.update(precision_metric.compute(predictions=predictions, references=labels))
    results.update(accuracy_metric.compute(predictions=predictions, references=labels))

    return results


dataset = DatasetDict(d)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42)


def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english", num_labels=2
    ).to("cuda")


training_args = TrainingArguments(
    output_dir="model-question-classification",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    report_to="wandb",
)

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

best_trial = trainer.hyperparameter_search(
    direction="maximize", backend="ray", n_trials=25, hp_space=hp_space
)

print(best_trial)

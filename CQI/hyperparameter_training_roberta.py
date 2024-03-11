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
    DebertaTokenizerFast,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)


def hp_space(trial):
    return {
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "per_device_train_batch_size": tune.choice([4, 5, 6, 7, 8, 9, 10, 11]),
        "num_train_epochs": tune.choice([3, 4, 5, 6]),
        "seed": tune.uniform(2, 42),
    }


torch.cuda.empty_cache()
gc.collect()

en_df = pd.read_csv("final_dataset_english.tsv", sep="\t")
output = en_df.groupby("category").apply(
    lambda group: group.sample(4938).reset_index(drop=True)
)
max_length = output["question"].apply(lambda x: len(x)).max()
train, test = train_test_split(output[["question", "category"]], test_size=0.01)

d = {
    "train": Dataset.from_dict(
        {
            "text": train["question"].values.tolist(),
            "label": train["category"].values.tolist(),
        }
    ),
    "test": Dataset.from_dict(
        {
            "text": test["question"].values.tolist(),
            "label": test["category"].values.tolist(),
        }
    ),
}

tokenizer = AutoTokenizer.from_pretrained("roberta-base", model_max_length=max_length)

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
    return RobertaForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=2
    ).to("cuda")


training_args = TrainingArguments(
    output_dir="model-question-classification",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
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

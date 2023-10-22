import pandas as pd
from sklearn.model_selection import train_test_split
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification, Trainer
import numpy as np
import evaluate
import torch
from ray import tune
import gc
import os

'''
Models use in trials: 

distilbert-base-uncased-finetuned-sst-2-english
distilbert-base-uncased
prajjwal1/bert-tiny
roberta-base
microsoft/deberta-base
'''

def hp_space(trial):
    return {

        "learning_rate": tune.loguniform(1e-6, 1e-4),

        "per_device_train_batch_size": tune.choice([8, 12, 16]),

        "num_train_epochs": tune.choice([3, 4, 5, 6]),

        "seed": tune.uniform(2, 42)

    }


torch.cuda.empty_cache()
gc.collect()

if not os.path.exists('train.csv') and not os.path.exists('test.csv') and not os.path.exists('validate.csv'):
    en_df = pd.read_csv("final_dataset_english.tsv", sep='\t')
    output = en_df.groupby('category').apply(lambda group: group.sample(4938).reset_index(drop=True))
    train, test = train_test_split(output[["question", "category"]], test_size=0.2,
                                        stratify=output[["category"]], random_state=42)

    test, val = train_test_split(test[["question", "category"]], test_size=0.5,
                                   stratify=test[["category"]], random_state=42)



    train.to_csv('train.csv', index=False, sep='\t')
    test.to_csv('test.csv', index=False, sep='\t')
    val.to_csv('validate.csv', index=False, sep='\t')
else:
    train = pd.read_csv("train.csv", sep='\t')
    val = pd.read_csv("validate.csv", sep='\t')
    test = pd.read_csv("test.csv", sep='\t')

d = {'train': Dataset.from_dict(
    {'text': train['question'].values.tolist(), 'label': train['category'].values.tolist()}),
    'test': Dataset.from_dict(
        {'text': test['question'].values.tolist(), 'label': test['category'].values.tolist()})
}

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", model_max_length=max_length)

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
    return AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english",
                                                              num_labels=2).to("cuda")


training_args = TrainingArguments(output_dir="model-question-classification", evaluation_strategy="epoch",
                                  num_train_epochs=3, per_device_train_batch_size=16)

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

best_trial = trainer.hyperparameter_search(
    direction="maximize",
    backend="ray",
    n_trials=50,
    hp_space=hp_space
)

print(best_trial)

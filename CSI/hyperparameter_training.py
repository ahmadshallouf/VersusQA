import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification, Trainer
import numpy as np
import evaluate
import torch
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
import gc
import os

CONFIG_PATH = ""


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


config = load_config("configuration.yaml")


def hp_space(trial):
    return {

        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "per_device_train_batch_size": tune.choice([8, 16]),
        "num_train_epochs": tune.choice([3, 4, 5, 6]),
        "weight_decay": tune.loguniform(1e-3, 1e-1),
        "warmup_steps": tune.uniform(0, 500),
        "seed": tune.uniform(2, 42)

    }


torch.cuda.empty_cache()
gc.collect()

os.environ["WANDB_PROJECT"] = "comparative-sentence-identification-hp-bert-tiny" # + config["model"]["name"]
# os.environ["WANDB_LOG_MODEL"] = "end"
os.environ["WANDB_WATCH"] = "all"
os.environ["WANDB_SILENT"] = "false"

en_df = pd.read_csv(f"""{config["data"]["folder_path"]}dataset_transformed.csv""", sep='\t')
max_length = en_df['x'].apply(lambda x: len(x)).max()

train = pd.read_csv("Dataset/train.csv", sep='\t')
val = pd.read_csv("Dataset/validate.csv", sep='\t')
test = pd.read_csv("Dataset/test.csv", sep='\t')

d = {'train': Dataset.from_dict(
    {'text': train['x'].values.tolist(), 'label': train['y'].values.tolist()}
),
    'validation': Dataset.from_dict(
        {'text': val['x'].values.tolist(), 'label': val['y'].values.tolist()}
    ),
    'test': Dataset.from_dict(
        {'text': test['x'].values.tolist(), 'label': test['y'].values.tolist()}
    )
}

tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], model_max_length=max_length)

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
    results.update(f1_metric.compute(predictions=predictions, references=labels, average='weighted'))
    results.update(recall_metric.compute(predictions=predictions, references=labels, average='weighted'))
    results.update(precision_metric.compute(predictions=predictions, references=labels, average='weighted'))
    results.update(accuracy_metric.compute(predictions=predictions, references=labels))

    return results


dataset = DatasetDict(d)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
small_eval_dataset = tokenized_datasets["validation"].shuffle(seed=42)
small_test_dataset = tokenized_datasets["test"].shuffle(seed=42)


def model_init():
    return AutoModelForSequenceClassification.from_pretrained(config["model"]["name"],
                                                              num_labels=3).to("cuda")


training_args = TrainingArguments(output_dir=config["train"]["checkpoint_path"],
                                  overwrite_output_dir=config["train"]["overwrite_checkpoint_path"],
                                  seed=config["seed"],
                                  data_seed=config["seed"],
                                  run_name=config["log"]["run_name"],
                                  load_best_model_at_end=config["model"]["load_best_at_end"],
                                  metric_for_best_model=config["model"]["metric_for_best"],
                                  evaluation_strategy=config["eval"]["strategy"])
training_args.set_dataloader(sampler_seed=config["seed"],
                             train_batch_size=config["train"]["batch_size"],
                             eval_batch_size=config["eval"]["batch_size"])  # auto_find_batch_size=True
training_args.set_evaluate(strategy=config["eval"]["strategy"], steps=config["eval"]["steps"], delay=config["eval"]["delay"],
                           batch_size=config["eval"]["batch_size"])
training_args.set_logging(strategy=config["log"]["strategy"], steps=config["log"]["steps"],
                          report_to=config["log"]["report_to"],
                          first_step=config["log"]["first_step"],
                          level=config["log"]["level"])
# args.set_lr_scheduler(name=config["lr_name"],
#                       warmup_steps=config["lr_warmup_steps"])
training_args.set_optimizer(name=config["optimizer_name"],
                            learning_rate=config["optimizer_learning_rate"])
# args.set_save(strategy=config["save"]["strategy"], steps=config["save"]["steps"])
# args.set_testing() # test_batch_size = eval
training_args.set_training(num_epochs=config["train"]["num_epochs"],
                           batch_size=config["train"]["batch_size"])

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

best_trial = trainer.hyperparameter_search(
    direction="maximize",
    backend="ray",
    n_trials=10,
    hp_space=hp_space,
    scheduler=PopulationBasedTraining(metric="objective", mode="max",
                                      hyperparam_mutations=hp_space("trial")),
)

print(best_trial)

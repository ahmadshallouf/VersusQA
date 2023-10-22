import pandas as pd
from sklearn.model_selection import train_test_split, KFold
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
import yaml

CONFIG_PATH = ""

F1_METRIC = evaluate.load("f1")
RECALL_METRIC = evaluate.load("recall")
PRECISION_METRIC = evaluate.load("precision")
ACCURACY_METRIC = evaluate.load("accuracy")


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
        # "warmup_steps": tune.uniform(0, 500),
        # "seed": tune.uniform(2, 42)

    }


def tokenize_function(examples):
    return tokenizer(examples["text"])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    results = {}
    results.update(F1_METRIC.compute(predictions=predictions, references=labels, average='weighted'))
    results.update(RECALL_METRIC.compute(predictions=predictions, references=labels, average='weighted'))
    results.update(PRECISION_METRIC.compute(predictions=predictions, references=labels, average='weighted'))
    results.update(ACCURACY_METRIC.compute(predictions=predictions, references=labels))

    return results


def model_init():
    return AutoModelForSequenceClassification.from_pretrained(config["model"]["name"],
                                                              num_labels=4).to("cuda")


def main():
    torch.cuda.empty_cache()
    gc.collect()

    os.environ["WANDB_PROJECT"] = "final-csi-" + config["model"]["name"]
    # os.environ["WANDB_LOG_MODEL"] = "end"
    os.environ["WANDB_WATCH"] = "all"
    os.environ["WANDB_SILENT"] = "false"

    train = pd.read_csv("Dataset/train.csv", sep='\t')
    val = pd.read_csv("Dataset/val.csv", sep='\t')
    test = pd.read_csv("Dataset/test.csv", sep='\t')

    d = {'train': Dataset.from_dict(
        {'text': train['x'].values.tolist(), 'label': train['y'].values.tolist()}
    ),
        'validation': Dataset.from_dict(
            {'text': val['x'].values.tolist(), 'label': val['y'].values.tolist()}
        ),
        'test_none': Dataset.from_dict(
            {'text': test[test['y'] == 0]['x'].values.tolist(), 'label': test[test['y'] == 0]['y'].values.tolist()}
        ),
        'test_better': Dataset.from_dict(
            {'text': test[test['y'] == 2]['x'].values.tolist(), 'label': test[test['y'] == 2]['y'].values.tolist()}
        ),
        'test_worse': Dataset.from_dict(
            {'text': test[test['y'] == 3]['x'].values.tolist(), 'label': test[test['y'] == 3]['y'].values.tolist()}
        ),
        'test': Dataset.from_dict(
            {'text': test['x'].values.tolist(), 'label': test['y'].values.tolist()}
        ),
    }

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], padding="max_length", truncation=True)

    dataset = DatasetDict(d)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(output_dir=f"./{config['log']['run_name']}-finetuned-obj/",
                                      overwrite_output_dir=f"./{config['log']['run_name']}-finetuned-obj/",
                                      seed=config["seed"],
                                      data_seed=config["seed"],
                                      run_name=config["log"]["run_name"],
                                      load_best_model_at_end=f"./{config['log']['run_name']}-best/",
                                      metric_for_best_model=config["model"]["metric_for_best"],
                                      evaluation_strategy=config["eval"]["strategy"])
    training_args.set_dataloader(sampler_seed=config["seed"],
                                 train_batch_size=config["train"]["batch_size"],
                                 eval_batch_size=config["eval"]["batch_size"])  # auto_find_batch_size=True
    training_args.set_evaluate(strategy=config["eval"]["strategy"], steps=config["eval"]["steps"],
                               delay=config["eval"]["delay"],
                               batch_size=config["eval"]["batch_size"])
    training_args.set_logging(strategy=config["log"]["strategy"], steps=config["log"]["steps"],
                              report_to=config["log"]["report_to"],
                              first_step=config["log"]["first_step"],
                              level=config["log"]["level"])
    # training_args.set_lr_scheduler(name=config["lr_name"],
    #                               warmup_steps=config["lr_warmup_steps"])
    training_args.set_optimizer(name=config["optimizer_name"],
                                learning_rate=config["optimizer_learning_rate"], )
    #                           weight_decay=config["lr_weight_decay"])
    # args.set_save(strategy=config["save"]["strategy"], steps=config["save"]["steps"])
    # args.set_testing() # test_batch_size = eval
    training_args.set_training(num_epochs=config["train"]["num_epochs"],
                               batch_size=config["train"]["batch_size"])

    model = model_init()

    trainer = Trainer(
        model=model,
        # model_init=model_init,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    # best_trial = trainer.hyperparameter_search(
    #     direction="maximize",
    #     backend="ray",
    #     n_trials=10,
    #     hp_space=hp_space,
    #     scheduler=PopulationBasedTraining(metric="objective", mode="max",
    #                                       hyperparam_mutations=hp_space("trial")),
    # )
    # print(best_trial)

    print("Training")
    trainer.train()

    print("Inference")
    results_file = open(f"{config['log']['run_name']}-final.txt", "w")
    results_file.write("Testing None\n")
    results = trainer.evaluate(eval_dataset=tokenized_datasets["test_none"])
    results_file.writelines([f"{results}", "\n"])

    results_file.write("Testing Better\n")
    results = trainer.evaluate(eval_dataset=tokenized_datasets["test_better"])
    results_file.writelines([f"{results}", "\n"])

    results_file.write("Testing Worse\n")
    results = trainer.evaluate(eval_dataset=tokenized_datasets["test_worse"])
    results_file.writelines([f"{results}", "\n"])

    results_file.write("Testing Whole\n")
    results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    results_file.writelines([f"{results}", "\n"])
    results_file.close()

    print("Save the model")
    model.save_pretrained(f"./{config['log']['run_name']}-best/")
    tokenizer.save_pretrained(f"./{config['log']['run_name']}-best/")

main()
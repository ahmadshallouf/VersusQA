import os

import pandas as pd
from datasets import Dataset
from datasets.dataset_dict import DatasetDict
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from utils_bert import (
    compute_metrics,
    compute_objective,
    load_config,
    model_init_helper,
    tokenize_function,
)

config = load_config("config.yaml")


def raytune_hp_space(trial):
    return {
        "learning_rate": tune.choice([1e-5, 3e-5, 5e-5, 7e-5, 1e-4]),
        "per_device_train_batch_size": tune.choice([8, 16]),
        "num_train_epochs": tune.randint(3, 20),
        "weight_decay": tune.choice([1e-4, 1e-3, 1e-2, 1e-1]),
        "warmup_steps": tune.choice([100, 200, 300, 400]),
    }


def optimize_bert():
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])

    # uncomment to set a fixed seed
    # transformers.set_seed(config["seed"])

    os.environ["WANDB_PROJECT"] = "optimize-sc-" + config["log"]["run_name"]
    os.environ["WANDB_LOG_MODEL"] = "end"
    os.environ["WANDB_WATCH"] = "all"
    os.environ["WANDB_SILENT"] = "false"

    train = pd.read_csv("data/train.csv", sep="\t")
    val = pd.read_csv("data/val.csv", sep="\t")
    test = pd.read_csv("data/test.csv", sep="\t")

    d = {
        "train": Dataset.from_dict(
            {"text": train["x"].values.tolist(), "label": train["y"].values.tolist()}
        ),
        "validation": Dataset.from_dict(
            {"text": val["x"].values.tolist(), "label": val["y"].values.tolist()}
        ),
        "test_none": Dataset.from_dict(
            {
                "text": test[test["y"] == 0]["x"].values.tolist(),
                "label": test[test["y"] == 0]["y"].values.tolist(),
            }
        ),
        "test_better": Dataset.from_dict(
            {
                "text": test[test["y"] == 2]["x"].values.tolist(),
                "label": test[test["y"] == 2]["y"].values.tolist(),
            }
        ),
        "test_worse": Dataset.from_dict(
            {
                "text": test[test["y"] == 3]["x"].values.tolist(),
                "label": test[test["y"] == 3]["y"].values.tolist(),
            }
        ),
        "test": Dataset.from_dict(
            {"text": test["x"].values.tolist(), "label": test["y"].values.tolist()}
        ),
    }

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"], padding="max_length", truncation=True
    )

    dataset = DatasetDict(d)

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
    )

    training_args = TrainingArguments(
        output_dir=f"./{config['log']['run_name']}-finetuned-obj/",
        overwrite_output_dir=f"./{config['log']['run_name']}-finetuned-obj/",
        # seed=config["seed"],              # uncomment to set a fixed seed
        # data_seed=config["seed"],         # uncomment to set a fixed seed
        run_name=config["log"]["run_name"],
        load_best_model_at_end=f"./{config['log']['run_name']}-best/",
        metric_for_best_model=config["model"]["metric_for_best"],
        evaluation_strategy=config["eval"]["strategy"],
    )
    training_args.set_dataloader(
        # sampler_seed=config["seed"],      # uncomment to set a fixed seed
        eval_batch_size=config["eval"]["batch_size"],
    )
    training_args.set_evaluate(
        strategy=config["eval"]["strategy"],
        steps=config["eval"]["steps"],
        delay=config["eval"]["delay"],
        batch_size=config["eval"]["batch_size"],
    )
    training_args.set_logging(
        strategy=config["log"]["strategy"],
        steps=config["log"]["steps"],
        report_to=config["log"]["report_to"],
        first_step=config["log"]["first_step"],
        level=config["log"]["level"],
    )
    training_args.set_lr_scheduler(
        name=config["learning_rate_scheduler"]["name"],
    )

    training_args.set_optimizer(
        name=config["optimizer"]["name"],
    )
    training_args.set_testing(batch_size=config["test"]["batch_size"])

    trainer = Trainer(
        model_init=model_init_helper(),
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    print("Hyperparameter Search")
    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="ray",
        hp_space=raytune_hp_space,
        compute_objective=compute_objective,
        n_trials=20,
        search_alg=HyperOptSearch(metric="objective", mode="max"),
        scheduler=ASHAScheduler(metric="objective", mode="max"),
        log_to_file=True,
    )

    print("Best trial:", best_trial)


if __name__ == "__main__":
    optimize_bert()

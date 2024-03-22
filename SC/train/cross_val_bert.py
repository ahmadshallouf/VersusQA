import os

import numpy as np
import pandas as pd
from datasets import Dataset
from datasets.dataset_dict import DatasetDict
from sklearn.model_selection import KFold
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from utils_bert import (
    compute_metrics,
    load_config,
    model_init_helper,
    tokenize_function,
)

config = load_config("config.yaml")


def cross_val_bert():
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])

    # uncomment to set a fixed seed
    # transformers.set_seed(config["seed"])

    os.environ["WANDB_PROJECT"] = "cv-train-cs-" + config["log"]["run_name"]
    os.environ["WANDB_LOG_MODEL"] = "end"
    os.environ["WANDB_WATCH"] = "all"
    os.environ["WANDB_SILENT"] = "false"

    train = pd.read_csv("data/train.csv", sep="\t")
    val = pd.read_csv("data/val.csv", sep="\t")
    train = pd.concat([train, val])
    test = pd.read_csv("data/test.csv", sep="\t")
    train = pd.concat([train, test])

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"], padding="max_length", truncation=True
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
        train_batch_size=config["train"]["batch_size"],
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
        warmup_steps=config["learning_rate_scheduler"]["warmup_steps"],
    )

    training_args.set_optimizer(
        name=config["optimizer"]["name"],
        learning_rate=config["optimizer"]["learning_rate"],
        weight_decay=config["optimizer"]["weight_decay"],
    )

    training_args.set_testing(batch_size=config["test"]["batch_size"])
    training_args.set_training(
        num_epochs=config["train"]["num_epochs"],
        batch_size=config["train"]["batch_size"],
    )

    print("K-Fold Cross Validation")
    if not os.path.exists("./results/"):
        os.mkdir("./results/")

    train_id = np.arange(len(train))
    kf = KFold(n_splits=10)

    split = 1
    results_df = pd.DataFrame()
    for train_split, test_split in kf.split(train_id):
        print(f"Split {split}")
        train_dataset = train.iloc[train_split, :]
        test_dataset = train.iloc[test_split, :]

        print(f"{len(train) = }, {len(train_dataset) = }, {len(test_dataset) = }")

        d = {
            "train": Dataset.from_dict(
                {
                    "text": train_dataset["x"].values.tolist(),
                    "label": train_dataset["y"].values.tolist(),
                }
            ),
            "test_none": Dataset.from_dict(
                {
                    "text": test_dataset[test_dataset["y"] == 0]["x"].values.tolist(),
                    "label": test_dataset[test_dataset["y"] == 0]["y"].values.tolist(),
                }
            ),
            "test_better": Dataset.from_dict(
                {
                    "text": test_dataset[test_dataset["y"] == 2]["x"].values.tolist(),
                    "label": test_dataset[test_dataset["y"] == 2]["y"].values.tolist(),
                }
            ),
            "test_worse": Dataset.from_dict(
                {
                    "text": test_dataset[test_dataset["y"] == 3]["x"].values.tolist(),
                    "label": test_dataset[test_dataset["y"] == 3]["y"].values.tolist(),
                }
            ),
            "test": Dataset.from_dict(
                {
                    "text": test_dataset["x"].values.tolist(),
                    "label": test_dataset["y"].values.tolist(),
                }
            ),
        }

        dataset = DatasetDict(d)

        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            fn_kwargs={"tokenizer": tokenizer},
        )

        trainer = Trainer(
            model=model_init_helper()(),
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
        )

        trainer.train()

        results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
        results.update({"dataset": "test"})
        results_df = pd.concat([results_df, pd.DataFrame([results])])

        results = trainer.evaluate(eval_dataset=tokenized_datasets["test_better"])
        results.update({"dataset": "test_better"})
        results_df = pd.concat([results_df, pd.DataFrame([results])])

        results = trainer.evaluate(eval_dataset=tokenized_datasets["test_worse"])
        results.update({"dataset": "test_worse"})
        results_df = pd.concat([results_df, pd.DataFrame([results])])

        results = trainer.evaluate(eval_dataset=tokenized_datasets["test_none"])
        results.update({"dataset": "test_none"})
        results_df = pd.concat([results_df, pd.DataFrame([results])])

        split += 1

    results_df.to_csv(
        f"./results/{config['log']['run_name']}-cv-sc.csv",
        index=False,
    )


if __name__ == "__main__":
    cross_val_bert()
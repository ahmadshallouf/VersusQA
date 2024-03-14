import os

import pandas as pd
import transformers
from datasets import Dataset
from datasets.dataset_dict import DatasetDict
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from utils_bert import compute_metrics, load_config, model_init, tokenize_function

config = load_config("config.yaml")


def main():
    transformers.set_seed(config["seed"])

    os.environ["WANDB_PROJECT"] = "draft-csi-" + config["log"]["run_name"]
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

    global tokenizer
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
        seed=config["seed"],
        data_seed=config["seed"],
        run_name=config["log"]["run_name"],
        load_best_model_at_end=f"./{config['log']['run_name']}-best/",
        metric_for_best_model=config["model"]["metric_for_best"],
        evaluation_strategy=config["eval"]["strategy"],
    )
    training_args.set_dataloader(
        sampler_seed=config["seed"],
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
    # args.set_save(strategy=config["save"]["strategy"], steps=config["save"]["steps"])
    training_args.set_testing(batch_size=config["test"]["batch_size"])
    training_args.set_training(
        num_epochs=config["train"]["num_epochs"],
        batch_size=config["train"]["batch_size"],
    )

    trainer = Trainer(
        model=model_init(),
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
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
    trainer.save_pretrained(f"./{config['log']['run_name']}-best/")


main()

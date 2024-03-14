import os

import transformers
from datasets import Dataset, DatasetDict
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from utils_bert import (
    compute_metrics,
    compute_objective,
    load_config,
    model_init_helper,
    read_data,
    tokenize_and_align_labels,
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
    transformers.set_seed(config["seed"])

    os.environ["WANDB_PROJECT"] = "optimize-oai-" + config["log"]["run_name"]
    os.environ["WANDB_LOG_MODEL"] = "end"
    os.environ["WANDB_WATCH"] = "all"
    os.environ["WANDB_SILENT"] = "false"

    """
    # Data loading
    # """

    train = read_data("train.csv")
    val = read_data("valid.csv")
    test = read_data("test.csv")

    ner_data = DatasetDict(
        {
            "train": Dataset.from_pandas(train),
            "valid": Dataset.from_pandas(val),
            "test": Dataset.from_pandas(test),
        }
    )

    """
    # Training
    """

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"],
        model_max_length=config["model"]["model_max_length"],
        add_prefix_space=True,
    )

    tokenized_datasets = ner_data.map(
        tokenize_and_align_labels,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    args = TrainingArguments(
        output_dir=f"./{config['log']['run_name']}-finetuned-obj/",
        overwrite_output_dir=f"./{config['log']['run_name']}-finetuned-obj/",
        seed=config["seed"],
        data_seed=config["seed"],
        run_name=config["log"]["run_name"],
        load_best_model_at_end=f"./{config['log']['run_name']}-best-end/",
        metric_for_best_model=config["model"]["metric_for_best"],
        evaluation_strategy=config["eval"]["strategy"],
        include_inputs_for_metrics=True,
    )
    args.set_dataloader(
        sampler_seed=config["seed"],
        # train_batch_size=config["train"]["batch_size"],
        eval_batch_size=config["eval"]["batch_size"],
    )
    args.set_evaluate(
        strategy=config["eval"]["strategy"],
        steps=config["eval"]["steps"],
        delay=config["eval"]["delay"],
        batch_size=config["eval"]["batch_size"],
    )
    args.set_logging(
        strategy=config["log"]["strategy"],
        steps=config["log"]["steps"],
        report_to=config["log"]["report_to"],
        first_step=config["log"]["first_step"],
        level=config["log"]["level"],
    )
    args.set_lr_scheduler(
        name=config["learning_rate_scheduler"]["name"],
        # warmup_steps=config["learning_rate_scheduler"]["warmup_steps"],
    )

    args.set_optimizer(
        name=config["optimizer"]["name"],
        # learning_rate=config["optimizer"]["learning_rate"],
        # weight_decay=config["optimizer"]["weight_decay"],
    )
    # args.set_save(strategy=config["save"]["strategy"], steps=config["save"]["steps"])
    args.set_testing(batch_size=config["test"]["batch_size"])
    # args.set_training(
    #     # num_epochs=config["train"]["num_epochs"],
    #     # batch_size=config["train"]["batch_size"],
    # )

    trainer = Trainer(
        model_init=model_init_helper(),
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Hyperparameter Search")
    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="ray",
        hp_space=raytune_hp_space,
        compute_objective=compute_objective,
        n_trials=1,
        search_alg=HyperOptSearch(metric="objective", mode="max"),
        scheduler=ASHAScheduler(metric="objective", mode="max"),
        log_to_file=True,
    )

    print("Best trial:", best_trial)


if __name__ == "__main__":
    optimize_bert()

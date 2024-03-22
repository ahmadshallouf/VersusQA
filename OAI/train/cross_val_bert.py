import os

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import KFold
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from utils_bert import (
    compute_metrics,
    load_config,
    model_init_helper,
    read_data,
    tokenize_and_align_labels,
)

config = load_config("config.yaml")


def cross_val_bert():
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])

    # uncomment to set a fixed seed
    # transformers.set_seed(config["seed"])

    os.environ["WANDB_PROJECT"] = "cv-train-oai-" + config["log"]["run_name"] + ""
    os.environ["WANDB_LOG_MODEL"] = "end"
    os.environ["WANDB_WATCH"] = "all"
    os.environ["WANDB_SILENT"] = "false"

    """
    # Data loading
    # """

    train = read_data("train.csv")
    val = read_data("valid.csv")
    train = pd.concat([train, val])
    test = read_data("test.csv")
    train = pd.concat([train, test])

    ner_data = DatasetDict(
        {
            "train": Dataset.from_pandas(train),
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
        # seed=config["seed"],  # uncomment to set a fixed seed
        # data_seed=config["seed"],  # uncomment to set a fixed seed
        run_name=config["log"]["run_name"],
        load_best_model_at_end=f"./{config['log']['run_name']}-best-end/",
        metric_for_best_model=config["model"]["metric_for_best"],
        evaluation_strategy=config["eval"]["strategy"],
        include_inputs_for_metrics=True,
    )
    args.set_dataloader(
        # sampler_seed=config["seed"],  # uncomment to set a fixed seed
        train_batch_size=config["train"]["batch_size"],
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
        warmup_steps=config["learning_rate_scheduler"]["warmup_steps"],
    )

    args.set_optimizer(
        name=config["optimizer"]["name"],
        learning_rate=config["optimizer"]["learning_rate"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    args.set_testing(batch_size=config["test"]["batch_size"])
    args.set_training(
        num_epochs=config["train"]["num_epochs"],
        batch_size=config["train"]["batch_size"],
    )

    print("K-Fold Cross Validation")
    if not os.path.exists("./results/"):
        os.mkdir("./results/")

    train_id = np.arange(len(tokenized_datasets["train"]))
    kf = KFold(n_splits=10)

    split = 1
    results_df = pd.DataFrame()
    for train, test in kf.split(train_id):
        print(f"Split {split}")
        train_dataset = tokenized_datasets["train"].select(train)
        test_dataset = tokenized_datasets["train"].select(test)

        trainer = Trainer(
            model=model_init_helper()(),
            args=args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        results = trainer.evaluate()
        results.update({"dataset": f"test_{split}"})

        results_df = pd.concat([results_df, pd.DataFrame([results])])
        split += 1

    results_df.to_csv(
        f"./results/{config['log']['run_name']}-cv-oai.csv",
        index=False,
    )


if __name__ == "__main__":
    cross_val_bert()

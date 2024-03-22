import os

import pandas as pd
import transformers
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from utils_bert import (
    compute_metrics,
    evaluate_dataset,
    load_config,
    model_init_helper,
    read_data,
    tokenize_and_align_labels,
)

config = load_config("config.yaml")


def train_bert():
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])

    # uncomment to set a fixed seed
    transformers.set_seed(config["seed"])

    os.environ["WANDB_PROJECT"] = "train-oai-" + config["log"]["run_name"]
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
        seed=config["seed"],  # uncomment for a fixed seed
        data_seed=config["seed"],  # uncomment for a fixed seed
        run_name=config["log"]["run_name"],
        load_best_model_at_end=f"./{config['log']['run_name']}-best-end/",
        metric_for_best_model=config["model"]["metric_for_best"],
        evaluation_strategy=config["eval"]["strategy"],
        include_inputs_for_metrics=True,
    )
    args.set_dataloader(
        sampler_seed=config["seed"],  # uncomment for a fixed seed
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
    args.set_save(strategy=config["save"]["strategy"], steps=config["save"]["steps"])
    args.set_testing(batch_size=config["test"]["batch_size"])
    args.set_training(
        num_epochs=config["train"]["num_epochs"],
        batch_size=config["train"]["batch_size"],
    )

    trainer = Trainer(
        model=model_init_helper()(),
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training")
    trainer.train()

    print("Inference")
    results = trainer.evaluate(eval_dataset=tokenized_datasets["train"])
    results.update({"dataset": "train"})
    print_header = False
    if not os.path.exists("./results/"):
        os.mkdir("./results/")
    if not os.path.exists(f"./results/{config['log']['run_name']}-oai.csv"):
        print_header = True
    results_df = pd.DataFrame([results])

    results = trainer.evaluate(eval_dataset=tokenized_datasets["valid"])
    results.update({"dataset": "valid"})
    results_df = pd.concat([results_df, pd.DataFrame([results])])

    results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    results.update({"dataset": "test"})
    results_df = pd.concat([results_df, pd.DataFrame([results])])

    results_df.to_csv(
        f"./results/{config['log']['run_name']}-oai.csv",
        mode="a",
        index=False,
        header=print_header,
    )

    evaluate_dataset(trainer, tokenized_datasets["train"], "train")
    evaluate_dataset(trainer, tokenized_datasets["valid"], "valid")
    evaluate_dataset(trainer, tokenized_datasets["test"], "test")

    print("Save the model")
    trainer.save_model(f"./{config['log']['run_name']}-best/")


if __name__ == "__main__":
    train_bert()

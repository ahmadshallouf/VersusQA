import os

import transformers
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from utils_bert import (
    compute_metrics_helper,
    load_config,
    model_init_helper,
    read_data,
    tokenize_and_align_labels,
)

config = load_config("configuration.yaml")


def train_bert():
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])
    transformers.set_seed(config["seed"])

    os.environ["WANDB_PROJECT"] = "draft-" + config["log"]["run_name"]
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
        train_batch_size=config["train"]["batch_size"],
        eval_batch_size=config["eval"]["batch_size"],
    )  # auto_find_batch_size=True
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
    # args.set_lr_scheduler(name=config["lr_name"],
    #                       warmup_steps=config["lr_warmup_steps"])

    args.set_optimizer(
        name=config["optimizer_name"],
        learning_rate=config["optimizer_learning_rate"],
    )
    #                  weight_decay=config["lr_weight_decay"])
    # args.set_save(strategy=config["save"]["strategy"], steps=config["save"]["steps"])
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
        compute_metrics=compute_metrics_helper(tokenizer),
    )

    print("Training")
    trainer.train()

    print("Inference")
    results_file = open(f"{config['log']['run_name']}-final.txt", "w")
    results_file.write("Training\n")
    results = trainer.evaluate(eval_dataset=tokenized_datasets["train"])
    results_file.writelines([f"{results}", "\n"])

    results_file.write("Validating\n")
    results = trainer.evaluate(eval_dataset=tokenized_datasets["valid"])
    results_file.writelines([f"{results}", "\n"])

    results_file.write("Testing\n")
    results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    results_file.writelines([f"{results}", "\n"])
    results_file.close()

    print("Save the model")
    trainer.save_model(f"./{config['log']['run_name']}-best/")


if __name__ == "__main__":
    train_bert()

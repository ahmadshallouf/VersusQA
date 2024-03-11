import wandb
from datasets import load_dataset, load_metric
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

run = wandb.init(settings=wandb.Settings(_service_wait=300))

model_name = "sshleifer/distilbart-cnn-6-6"  # BART
# model_name = "google/pegasus-xsum"  # Pegasus
# model_name = "csebuetnlp/mT5_multilingual_XLSum"  # T5

tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset(
    "json",
    data_files={
        "train": "data-prep/train.json",
        "validation": "data-prep/val.json",
        "test": "data-prep/test.json",
    },
)
metric = load_metric("rouge")


def encode(example_batch):
    input_encodings = tokenizer(
        example_batch["input"], max_length=1024, truncation=True
    )

    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(
            example_batch["output"], max_length=256, truncation=True
        )

    return {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": target_encodings["input_ids"],
    }


encoded_dataset = dataset.map(encode, batched=True)
columns = ["input_ids", "labels", "attention_mask"]
encoded_dataset.set_format(type="torch", columns=columns)
seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer)


def model_init():
    return BartForConditionalGeneration.from_pretrained(model_name, return_dict=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    "test", evaluation_strategy="steps", eval_steps=500, disable_tqdm=True
)
trainer = Trainer(
    args=training_args,
    tokenizer=tokenizer,
    data_collator=seq2seq_data_collator,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    model_init=model_init,
    compute_metrics=compute_metrics,
)

trainer.hyperparameter_search(
    direction="maximize",
    backend="ray",
    n_trials=10,
    search_alg=HyperOptSearch(metric="objective", mode="max"),
    scheduler=ASHAScheduler(metric="objective", mode="max"),
)

run.finish()

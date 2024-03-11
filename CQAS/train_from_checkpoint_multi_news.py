from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

dataset = load_dataset("multi_news")

model_ckpt = "sshleifer/distilbart-cnn-6-6"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = BartForConditionalGeneration.from_pretrained(model_ckpt)

d_len = [len(tokenizer.encode(s)) for s in dataset["validation"]["document"]]
s_len = [len(tokenizer.encode(s)) for s in dataset["validation"]["summary"]]


def convert_examples_to_features(example_batch):
    input_encodings = tokenizer(
        example_batch["document"], max_length=1024, truncation=True
    )

    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(
            example_batch["summary"], max_length=256, truncation=True
        )

    return {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": target_encodings["input_ids"],
    }


dataset_tf = dataset.map(convert_examples_to_features, batched=True)

columns = ["input_ids", "labels", "attention_mask"]
dataset_tf.set_format(type="torch", columns=columns)
seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = TrainingArguments(
    output_dir="bart-multi-news",
    num_train_epochs=1,
    warmup_steps=500,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    logging_steps=10,
    push_to_hub=False,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    gradient_accumulation_steps=16,
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=seq2seq_data_collator,
    train_dataset=dataset_tf["train"],
    eval_dataset=dataset_tf["validation"],
)


# https://github.com/huggingface/transformers/issues/7198
trainer.train("bart-multi-news/")

from datasets import load_dataset
from transformers import BartForConditionalGeneration, AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer
import time


model_name = "sshleifer/distilbart-cnn-6-6"
output_path = "output/bart"

dataset = load_dataset('json', data_files={'train': 'data-prep/train.json',
                                           'validation': 'data-prep/val.json',
                                           'test': 'data-prep/test.json'})

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)


def convert_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch["input"], max_length=1024, truncation=True)

    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(example_batch["output"], max_length=256, truncation=True)

    return {"input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "labels": target_encodings["input_ids"]}

dataset_tf = dataset.map(convert_examples_to_features, batched=True)

columns = ["input_ids", "labels", "attention_mask"]
dataset_tf.set_format(type="torch", columns=columns)
seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = TrainingArguments(output_dir=output_path, num_train_epochs=8, warmup_steps=500,
                                  per_device_train_batch_size=1, per_device_eval_batch_size=1,
                                  weight_decay=0.01, logging_steps=10, push_to_hub=False,
                                  evaluation_strategy='steps', eval_steps=100, save_steps=50,
                                  gradient_accumulation_steps=16)

trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer,
                  data_collator=seq2seq_data_collator,
                  train_dataset=dataset_tf["train"],
                  eval_dataset=dataset_tf["validation"])

t0 = time.time()
trainer.train()
training_time = time.time()-t0

with open("performance.txt", "a") as file:
    file.write(f"Training time for {model_name}: {training_time}\n")



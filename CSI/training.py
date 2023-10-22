import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification, Trainer
import numpy as np
import evaluate
import os.path

'''
Models use in trials:

distilbert-base-uncased-finetuned-sst-2-english
distilbert-base-uncased
prajjwal1/bert-tiny
roberta-base
microsoft/deberta-base
'''

tokenizer = 'distilbert-base-uncased'#-finetuned-sst-2-english'
model = 'distilbert-base-uncased'#-finetuned-sst-2-english'

train = pd.read_csv("Dataset/train.csv", sep='\t')
val = pd.read_csv("Dataset/validate.csv", sep='\t')
test = pd.read_csv("Dataset/test.csv", sep='\t')

d = {'train': Dataset.from_dict(
    {'text': train['x'].values.tolist(), 'label': train['y'].values.tolist()}
    ),
    'validation': Dataset.from_dict(
        {'text': val['x'].values.tolist(), 'label': val['y'].values.tolist()}
    ),
    'test': Dataset.from_dict(
        {'text': test['x'].values.tolist(), 'label': test['y'].values.tolist()}
    )
}

tokenizer = AutoTokenizer.from_pretrained(tokenizer, padding="max_length", truncation=True)

f1_metric = evaluate.load("f1")
recall_metric = evaluate.load("recall")
precision_metric = evaluate.load("precision")
accuracy_metric = evaluate.load("accuracy")


def tokenize_function(examples):
    return tokenizer(examples["text"])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    results = {}
    results.update(f1_metric.compute(predictions=predictions, references=labels, average='weighted'))
    results.update(recall_metric.compute(predictions=predictions, references=labels, average='weighted'))
    results.update(precision_metric.compute(predictions=predictions, references=labels, average='weighted'))
    results.update(accuracy_metric.compute(predictions=predictions, references=labels))

    return results


dataset = DatasetDict(d)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
small_eval_dataset = tokenized_datasets["validation"].shuffle(seed=42)
small_test_dataset = tokenized_datasets["test"].shuffle(seed=42)

model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=3)

training_args = TrainingArguments(
    output_dir="model-question-classification",
    evaluation_strategy="epoch",
    learning_rate=3e-05,
    num_train_epochs=10,
    per_device_train_batch_size=12,
    seed=38
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

print('---------------------------------------------------------------------------------------------')

# evaluate the model on the test set and print results
pred = trainer.predict(small_test_dataset)
print(pred.metrics)

trainer.save_model("model_binary_classifier")
tokenizer.save_pretrained("model_binary_classifier")

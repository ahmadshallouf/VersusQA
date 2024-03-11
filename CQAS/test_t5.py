import json
import time

import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, MT5ForConditionalGeneration, T5Tokenizer


def test_model_performance(model_path, cpu: bool):
    dataset = load_dataset(
        "json",
        data_files={
            "train": "data-prep/train.json",
            "validation": "data-prep/val.json",
            "test": "data-prep/test.json",
        },
    )

    input_examples = dataset["test"]["input"]
    references = dataset["test"]["output"]

    tokenizer = T5Tokenizer.from_pretrained(model_path)

    model = MT5ForConditionalGeneration.from_pretrained(model_path)
    if cpu:
        input_ids = tokenizer(
            input_examples,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to("cpu")
    else:
        input_ids = tokenizer(
            input_examples,
            max_length=1024,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    summaries = model.generate(
        input_ids=input_ids["input_ids"],
        attention_mask=input_ids["attention_mask"],
        max_length=256,
    )

    counter = 1
    predictions = []
    t0 = time.time()
    for s in summaries:
        prediction = tokenizer.decode(
            s, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        print(f"prediction {counter}, {time.time()}")
        predictions.append(prediction)
        counter = counter + 1
    testing_time = time.time() - t0

    # Calc rouge scores
    rouge = evaluate.load("rouge")

    print("Starting rouge")
    results = rouge.compute(predictions=predictions, references=references)
    print("Ended rouge")

    with open("performance.txt", "a") as file:
        file.write(model_path)
        file.write(f"rouge1: {results['rouge1']}\n")
        file.write(f"rouge2: {results['rouge2']}\n")
        file.write(f"rougeL: {results['rougeL']}\n")
        file.write(f"rougeLsum: {results['rougeLsum']}\n")
        file.write(f"Testing time: {testing_time}\n\n")
        file.write("\n")

    print(results)


# test_model_performance("output/t5/", True)  # Fine-tuned t5
# test_model_performance("google/mt5-small", True)  # Baseline t5
test_model_performance("csebuetnlp/mT5_multilingual_XLSum", True)

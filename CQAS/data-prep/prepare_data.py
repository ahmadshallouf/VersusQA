import json
from datasets import load_dataset
import pandas as pd
import codecs
import re


def prepare_data_unprocessed():
    with open('chatgpt_dataset_few_shot_final.jsonlines', 'r') as json_file:
        str_json = json_file.read()
    json_data = json.loads(str_json)

    json_list = []
    for example in json_data:
        input = example["input"]
        outputs = example["outputs"]

        for output in outputs:
            json_list.append({
                "input": input,
                "output": output
            })

    total_samples = len(json_list)
    train_size = int(total_samples * 0.8)
    test_size = int(total_samples * 0.1)
    train_set = json_list[:train_size]
    test_set = json_list[train_size: train_size + test_size]
    val_set = json_list[train_size + test_size:]

    with open("train.jsonlines", "w") as outfile:
        outfile.write(json.dumps(train_set))

    with open("val.jsonlines", "w") as outfile:
        outfile.write(json.dumps(val_set))

    with open("test.jsonlines", "w") as outfile:
        outfile.write(json.dumps(test_set))


def prepare_data_huggingface(json_path, output_path):
    with open(json_path, 'r') as file:
        str = file.read()
        data = json.loads(str)

    lst = []
    for example in data:
        input = example["input"]
        output = example["output"]

        # Process input
        input = input.split("Summarize only relevant arguments from the list.\n\n")[1]
        input = input.split("\n\nAfter the summary, list the arguments you used below the text.")[0]
        input = input.split("\n")
        if input == ['']:
            continue
        arguments = []
        for argument in input:
            argument = re.split(r"\D", argument, maxsplit=1)[1].strip()
            arguments.append(argument)
        input = "\n".join(arguments)

        # Process output
        output = output.split("\n\nArguments used:")[0]
        output = re.sub(r'[\d]+', '', output)
        output = output.replace(" []", "")
        output = output.replace("[]", "")

        # Add to list
        lst.append({
            "input": "Summarize: " + input,
            "output": output
        })

    with open(output_path, "w") as outfile:
        for pair in lst:
            outfile.write(json.dumps(pair)+"\n")


# prepare_data_unprocessed()
prepare_data_huggingface("train.jsonlines", "train.json")
prepare_data_huggingface("val.jsonlines", "val.json")
prepare_data_huggingface("test.jsonlines", "test.json")

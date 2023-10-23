import os
from typing import Dict, List

import yaml
from fastapi import FastAPI

from transformers import pipeline

CONFIG_PATH = ""


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


config = load_config("configuration.yaml")

# ====================== API ==========================

app = FastAPI()  # Running on port 8000

if not os.path.exists('models'):
    os.makedirs('models')

if not os.path.exists('models/best'):
    os.makedirs('models/best')

if not os.path.exists('models/fast'):
    os.makedirs('models/fast')

if not os.listdir('models/best'):
    print("Downloading best model...")
    token_classifier = pipeline(
        "token-classification", model=config["data"]["last_model_path"],
        aggregation_strategy=config["test"]["pipeline_aggregation_strategy"])
    token_classifier.save_pretrained('models/best')
else:
    print("Loading best model...")
    token_classifier = pipeline(
        "token-classification", model='models/best',
        aggregation_strategy=config["test"]["pipeline_aggregation_strategy"])

if not os.listdir('models/fast'):
    print("Downloading fast model...")
    token_classifier_fast = pipeline(
        "token-classification", model=config["data"]["fast_model_path"],
        aggregation_strategy=config["test"]["pipeline_aggregation_strategy"])
    token_classifier_fast.save_pretrained('models/fast')
else:
    print("Loading fast model...")
    token_classifier_fast = pipeline(
        "token-classification", model='models/fast',
        aggregation_strategy=config["test"]["pipeline_aggregation_strategy"])


@app.get("/")
async def root():
    return "Welcome to Object and Aspect identification API, use /get_objects_and_aspects/{question} to proceed!"


@app.get("/get_objects_and_aspects/{fast}/{question}")
async def get_objects_and_aspects(question: str, fast: bool) -> Dict[str, List[str]]:
    return predict(question, fast)


def predict(question: str, fast: bool) -> Dict[str, List[str]]:
    print(f"This question will be analyzed: {question}")

    if fast:
        tokens = token_classifier_fast(question)
    else:
        tokens = token_classifier(question)

    result = {"objects": [token["word"] for token in tokens if token["entity_group"] == "OBJ"],
              "aspects": [token["word"] for token in tokens if token["entity_group"] == "ASP"]}

    return result

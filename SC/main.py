import os

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ====================== API ==========================
app = FastAPI()  # Create the API on Port 8000

if not os.path.exists("model"):
    os.makedirs("model")

if not os.listdir("model"):
    print("Downloading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "uhhlt/stance-comp-classifier", num_labels=3
    )  # .to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("uhhlt/stance-comp-classifier")
    model.save_pretrained("model")
    tokenizer.save_pretrained("model")
else:
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained("model", num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained("model")


class argument(BaseModel):
    value: str
    source: str


class request(BaseModel):
    object1: str
    object2: str
    arguments: list[argument]


class response(BaseModel):
    arguments1: list[argument]
    arguments2: list[argument]


@app.get("/")
async def root():
    return (
        "Welcome to Comparative Sentence Identification Machine!"
        "\nUse /is_comparative/{sentence} to check your Sentence."
    )


@app.get("/is_comparative/{sentence}")
async def is_comparative(sentence: str):
    return analyse_sentence(sentence)


@app.post("/get_arguments")
async def get_arguments(item: request):
    # concatenate object1, object2 with every argument in item
    print(
        "Received request with: "
        + str(len(item.arguments))
        + " arguments for "
        + item.object1
        + " and "
        + item.object2
        + "."
    )
    conc_arguments = []
    shadow_arguments = []
    arguments1 = []
    arguments2 = []

    for arg in item.arguments:
        obj1 = item.object1
        obj2 = item.object2

        # get index of obj1 inside arg.value not case sensitive
        index1 = arg.value.lower().find(obj1.lower())
        # get index of obj2 inside arg.value not case sensitive
        index2 = arg.value.lower().find(obj2.lower())

        # if obj1 is not in arg.value
        if index1 == -1:
            continue
        # if obj2 is not in arg.value
        if index2 == -1:
            continue

        # if obj1 is before obj2
        if index1 < index2:
            new_value = item.object1 + " [SEP] " + item.object2 + " [SEP] " + arg.value
            conc_arguments.append([False, argument(value=new_value, source=arg.source)])
            shadow_arguments.append(argument(value=arg.value, source=arg.source))
        # if obj2 is before obj1
        else:
            new_value = item.object2 + " [SEP] " + item.object1 + " [SEP] " + arg.value
            conc_arguments.append([True, argument(value=new_value, source=arg.source)])
            shadow_arguments.append(argument(value=arg.value, source=arg.source))

    # use analyse_sentence to get the arguments
    for i in range(len(conc_arguments)):
        cl = analyse_sentence(conc_arguments[i][1].value)

        if cl == 2 and conc_arguments[i][0] is False:
            arguments1.append(shadow_arguments[i])
        elif cl == 1 and conc_arguments[i][0] is False:
            arguments2.append(shadow_arguments[i])
        elif cl == 2 and conc_arguments[i][0] is True:
            arguments2.append(shadow_arguments[i])
        elif cl == 1 and conc_arguments[i][0] is True:
            arguments1.append(shadow_arguments[i])
    print("args1: " + str(arguments1) + " args2: " + str(arguments2))
    return response(arguments1=arguments1, arguments2=arguments2)


# ====================== ML ==========================


def analyse_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")  # .to("cuda")

    with torch.no_grad():
        logits = model(**inputs).logits
        print(logits)
        predicted_class_id = logits.argmax().item()

    return predicted_class_id

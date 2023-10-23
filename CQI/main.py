from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# ====================== API ==========================
app = FastAPI()  # Create the API on Port 8000

# if folder model is empty, download the model
print(os.listdir('model'))
if not os.listdir('model'):
    print("Downloading model...")
    model = AutoModelForSequenceClassification.from_pretrained("uhhlt/binary-compqa-classifier", num_labels=2)  # .to("cuda")
    tokenizer = AutoTokenizer.from_pretrained('uhhlt/binary-compqa-classifier')
    model.save_pretrained('model')
    tokenizer.save_pretrained('model')
else:
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained("model", num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained('model')



@app.get("/")
async def root():
    return "Welcome to Comparative Question Identification Machine!" \
           "\nUse /is_comparative/{question} to check your questions."


@app.get("/is_comparative/{question}")
async def is_comparative(question: str):
    return analyse_sentence(question)


# ====================== ML ==========================


def analyse_sentence(sentence):
    print("This sentence will be analyzed: " + sentence)

    inputs = tokenizer(sentence, return_tensors="pt")  # .to("cuda")

    with torch.no_grad():
        logits = model(**inputs).logits
        print(logits)
        predicted_class_id = logits.argmax().item()

    return predicted_class_id == 1

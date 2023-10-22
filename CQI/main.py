from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ====================== API ==========================
app = FastAPI()  # Create the API on Port 8000

model = AutoModelForSequenceClassification.from_pretrained("model_binary_classifier", num_labels=2)  # .to("cuda")

tokenizer = AutoTokenizer.from_pretrained('./model_binary_classifier/')


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

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("uhhlt/stance-comp-classifier", num_labels=3)  # .to("cuda")
tokenizer = AutoTokenizer.from_pretrained('uhhlt/stance-comp-classifier')


def test(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")  # .to("cuda")

    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class_id = logits.argmax().item()

    if predicted_class_id == 0:
        return f'Object one is better than Object two: {logits}'
    elif predicted_class_id == 1:
        return f'The sentence is not comparative: {logits}'
    return f'Object two is better than Object one: {logits}'


demo = gr.Interface(fn=test, inputs="text", outputs="text")

if __name__ == "__main__":
    demo.launch(share=True)

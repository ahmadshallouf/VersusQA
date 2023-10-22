import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("uhhlt/binary-compqa-classifier",
                                                           num_labels=2)  # .to("cuda")
tokenizer = AutoTokenizer.from_pretrained('uhhlt/binary-compqa-classifier')

def test(sentence):

    inputs = tokenizer(sentence, return_tensors="pt")  # .to("cuda")

    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class_id = logits.argmax().item()

    if predicted_class_id == 1:
        return f'It is comparative. Resulting Tensor: {logits}'
    return f'It is NOT comparative. Resulting Tensor: {logits}'


demo = gr.Interface(fn=test, inputs="text", outputs="text")

if __name__ == "__main__":
    demo.launch(share=True)

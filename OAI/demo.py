import os

import yaml

import gradio as gr
from main import predict

CONFIG_PATH = ""


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


config = load_config("configuration.yaml")


# ====================== DEMO ==========================

def test(question: str) -> str:
    result = predict(question)

    result_string = f"""The found object(s): {', '.join(obj for obj in result["objects"])}\n"""
    result_string += f"""The found aspect(s): {', '.join(obj for obj in result["aspects"])}"""

    return result_string


demo = gr.Interface(fn=test, inputs="text", outputs="text")

if __name__ == "__main__":
    demo.launch(share=True)

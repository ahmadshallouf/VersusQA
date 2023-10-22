import gradio as gr
from summarizer import generate_summary


async def generate(object1, object2, arguments):
    arguments = arguments.splitlines()
    result = await generate_summary(object1, object2, arguments)
    return result


demo = gr.Interface(
    fn=generate,
    inputs=[gr.inputs.Textbox(lines=1, label="Object 1"),
            gr.inputs.Textbox(lines=1, label="Object 2"),
            gr.inputs.Textbox(lines=5, label="Input Text")
            ],
    outputs=gr.outputs.Textbox(label="Generated Summary")
)

demo.launch()

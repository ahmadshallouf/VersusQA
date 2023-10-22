import json
from transformers import BartForConditionalGeneration, AutoTokenizer
from pydantic import BaseModel
import time


def generate_summary(object1: str, object2: str, arguments: list[str]) -> str:
    prompt = "Summarize: " + "\n".join(arguments)

    tokenizer = AutoTokenizer.from_pretrained("../output/bart/checkpoint-400/")

    model = BartForConditionalGeneration.from_pretrained("../output/bart/checkpoint-400/")

    device = 'cpu'
    input_ids = tokenizer(prompt, max_length=1024, truncation=True, padding='max_length', return_tensors='pt').to(
        device)
    summaries = model.generate(input_ids=input_ids['input_ids'],
                               attention_mask=input_ids['attention_mask'],
                               max_length=256)
    decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=True)
                         for s in summaries]
    return decoded_summaries[0]


with open('cam.json', 'r') as json_file:
    str_json = json_file.read()
    json_data = json.loads(str_json)

object1 = json_data["object1"]["name"]
object2 = json_data["object2"]["name"]

arguments = []
for argument in json_data["object1"]["sentences"]:
    arguments.append(argument["text"])

for argument in json_data["object2"]["sentences"]:
    arguments.append(argument["text"])

t0 = time.time()
print(generate_summary(object1, object2, arguments))
testing_time = time.time()-t0
print(testing_time)
# 25.50338363647461
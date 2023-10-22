from fastapi import FastAPI
from pydantic import BaseModel
from summarizer import generate_summary
from transformers import BartForConditionalGeneration, AutoTokenizer

app = FastAPI()  # Running on port 8002
tokenizer = AutoTokenizer.from_pretrained("production-model")
model = BartForConditionalGeneration.from_pretrained("production-model")

# ====================== Models ==========================

class Item(BaseModel):
    object1: str
    object2: str
    arguments: list[str]

async def generate_summary(object1: str, object2: str, arguments: list[str]) -> str:
    prompt = "Summarize: " + "\n".join(arguments)
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



# ====================== API ==========================


@app.get("/")
async def root():
    return {"message": "BART multi-doc summarizer"}


@app.post("/summary")
async def summary(item: Item):
    return await generate_summary(item.object1, item.object2, item.arguments)

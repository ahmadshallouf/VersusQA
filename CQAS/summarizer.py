from pydantic import BaseModel
from transformers import AutoTokenizer, BartForConditionalGeneration


async def generate_summary(object1: str, object2: str, arguments: list[str]) -> str:
    prompt = "Summarize: " + "\n".join(arguments)

    tokenizer = AutoTokenizer.from_pretrained("uhhlt/comp-summarization-distilbart-cnn")
    model = BartForConditionalGeneration.from_pretrained(
        "uhhlt/comp-summarization-distilbart-cnn"
    )

    device = "cpu"
    input_ids = tokenizer(
        prompt,
        max_length=1024,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(device)
    summaries = model.generate(
        input_ids=input_ids["input_ids"],
        attention_mask=input_ids["attention_mask"],
        max_length=256,
    )
    decoded_summaries = [
        tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for s in summaries
    ]
    return decoded_summaries[0]

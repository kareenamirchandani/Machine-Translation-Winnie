from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from datasets import load_dataset

raw = load_dataset("Sunbird/salt-dataset")

print(raw["train"]["English"][5])

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")

src_text = raw["train"]["English"][100]
# tgt_text = "Şeful ONU declară că nu există o soluţie militară în Siria"

model_inputs = tokenizer(src_text, return_tensors="pt")

print(model_inputs)

outputs = model(**model_inputs)

model_inputs = tokenizer(src_text, return_tensors="pt")

# translate from English to Hindi
generated_tokens = model.generate(
    **model_inputs,
    forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"]
)

print(generated_tokens)

translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

print(translated)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

# Loads a line of English from SALT dataset
raw = load_dataset("Sunbird/salt-dataset")
src_text = raw["train"]["English"][5]

# Change ">>teo<<" to other language codes to change output langauge"
# Without language code I think it defaults to lugbara (guess >>lug<< ??)
src_text = ">>teo<<" + src_text


# Gets model and tokenizer from sunbird
tokenizer = AutoTokenizer.from_pretrained("Sunbird/sunbird-en-mul")
model = AutoModelForSeq2SeqLM.from_pretrained("Sunbird/sunbird-en-mul")

# Converts string into sequence of token IDs
model_inputs = tokenizer(src_text, return_tensors="pt")

# Feeds input into model giving output as token IDs
model_outputs = model.generate(**model_inputs)

print(model_outputs)

#Convert token IDs back into string
out_text = tokenizer.batch_decode(model_outputs, skip_special_tokens=True)

# prints input and output sentence
print(src_text)
print(out_text)

# for i in range(1,1000,50):
#     print(tokenizer.convert_ids_to_tokens(i))

# print(tokenizer.convert_tokens_to_ids(">>teo<<"))
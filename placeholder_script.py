import re
import requests
named_entities = ["New York", "John Smith", "Microsoft"]
non_english_words = ["bonjour", "hola", "こんにちは"]
opus_mt_endpoint = "https://api.opusmt.com/translate"

text_to_translate = "This is a test sentence containing New York and bonjour."

named_entity_placeholder = "[NE]"
non_english_placeholder = "[NON-ENGLISH]"

for ne in named_entities:
    text_to_translate = re.sub(re.escape(ne), named_entity_placeholder, text_to_translate)

for non_eng in non_english_words:
    text_to_translate = re.sub(re.escape(non_eng), non_english_placeholder, text_to_translate)

params = {
    "src": "en",
    "trg": "es",
    "input": text_to_translate
}
response = requests.get(opus_mt_endpoint, params=params)
print(response.json()["output"])

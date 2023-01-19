## Do Same as dataAnalysis and overlap but by tokens instead of words
## Tokens from sunbird mul-en tokenizer
## Can test with other tokenizsers e.g. from afriBERT, mBART etc

from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained("Sunbird/sunbird-en-mul")

## Create list of all the tokens
tokenList = []
ID = 0
while True:
    token = tokenizer.convert_ids_to_tokens(ID)
    if ID == 1:
        tokenList.append("<unk>")
        ID += 1
        continue
    elif token == "<unk>":
        break
    else:
        tokenList.append(token)
        ID += 1

## Tokenize SALT dataset
dataset = load_dataset("Sunbird/salt-dataset", split="train")
languages = dataset.column_names
tokenized_data = {}
tokenData = {}
usedTokenSet = {}
unusedTokenSet = {}
tokenCount = {lang: {} for lang in languages}
tokenCountArr = {}
ranked = {}

idx = 1
for lang in languages:
    print(1)
    tokenized_data[lang] = dataset.map(lambda examples: tokenizer(examples[lang]), batched=True)
    tokenData[lang] = [token for example in tokenized_data[lang] for token in example["input_ids"]]
    usedTokenSet[lang] = set(tokenData[lang])
    unusedTokenSet[lang] = set(tokenList) - usedTokenSet[lang]
    for token in usedTokenSet[lang]:
        tokenCount[lang][token] = tokenData[lang].count(token)
    tokenCountArr[lang] = np.array([[key, value] for key, value in tokenCount[lang].items()], dtype = int)
    ranked[lang] = tokenCountArr[lang][tokenCountArr[lang][:,1].argsort()]
    ranked[lang] = ranked[lang][::-1,:]
    topTokens = 10
    print(2)
    plt.subplot(2,3, idx)
    idx += 1
    plt.plot(ranked[lang][0:topTokens, 1])
    plt.ylabel("Frequency")
    plt.xlabel("Token ID")
    plt.title(lang)
    plt.xticks(np.array(range(0,topTokens)), ranked[lang][0:topTokens, 0])
    print(3)
plt.show()







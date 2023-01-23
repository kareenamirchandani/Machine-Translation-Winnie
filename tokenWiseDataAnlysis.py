## Do Same as dataAnalysis and overlap but by tokens instead of words
## Tokens from sunbird mul-en tokenizer
## Can test with other tokenizsers e.g. from afriBERT, mBART etc


####### Issues with unused tokens !!!!!!! Don't trust !!!! Figures still work well

from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained("Davlan/afro-xlmr-large")

## Create list of all the tokens
tokenList = []
IDList = []
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
        tokenList.append(ID)
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
total = {}

idx = 1
print("Afro-XLMR-large")
for lang in languages:
    tokenized_data[lang] = dataset.map(lambda examples: tokenizer(examples[lang]), batched=True)
    tokenData[lang] = [token for example in tokenized_data[lang] for token in example["input_ids"]]
    usedTokenSet[lang] = set(tokenData[lang])
    unusedTokenSet[lang] = set(IDList) - usedTokenSet[lang]
    if 3 in unusedTokenSet:
        print("No UNK")
    for token in usedTokenSet[lang]:
        if token == 3:
            print("UNK token")
        tokenCount[lang][token] = tokenData[lang].count(token)
    tokenCountArr[lang] = np.array([[key, value] for key, value in tokenCount[lang].items()], dtype = int)
    total[lang] = np.sum(tokenCountArr[lang][:,1])
    ranked[lang] = tokenCountArr[lang][tokenCountArr[lang][:,1].argsort()]
    ranked[lang] = ranked[lang][::-1,:]
    topTokens = 10
    plt.subplot(2,3, idx)
    idx += 1
    plt.plot(ranked[lang][0:topTokens, 1])
    plt.ylabel("Frequency")
    plt.xlabel("Token ID")
    plt.title(lang)
    plt.xticks(np.array(range(0,topTokens)), ranked[lang][0:topTokens, 0])
    print(lang)
    print("\% of tokens Used: " + str(len(usedTokenSet[lang])/(len(unusedTokenSet[lang]) + len(usedTokenSet[lang]))))
    print("Num UNK: " + tokenCount[lang][tokenizer.convert_tokens_to_ids("<unk>")])
    print("Total Tokens: " + total[lang] )
    # How many top tokens needed to represent 75% of data
    topPercent = 0.75
    occurences = 0
    for i in range(0, np.shape(tokenCountArr[lang])[0]):
        if occurences > (topPercent * total[lang]):
            numTokens = i
            break
        occurences = occurences + tokenCountArr[lang][i, 1]
    
    print(str(numTokens) + " needed to represent " + str(topPercent*100) + "\% of data")

    print(3)
plt.show()







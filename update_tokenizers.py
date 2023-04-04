from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained("Davlan/afro-xlmr-mini")
pretrained_vocab = tokenizer.get_vocab()
d_swap = {v: k for k, v in pretrained_vocab.items()}
pretrained_tokens = list(pretrained_vocab.values())

dataset = load_dataset("Sunbird/salt-dataset", split="train")
languages = dataset.column_names

subword_dict = {}
tokenized_data = {}

# loop through the languages in the dataset
for lang in languages:
    # tokenize salt
    tokenized_data = dataset.map(lambda examples: tokenizer(examples[lang]), batched=True)
    # loop through the tokenized text data 
    for tokens in tokenized_data:
        subwords = tokens["input_ids"]
        # add the subwords for each token to the subword dictionary 
        for i in range(len(subwords)):
            token = subwords[i]
            try:
                subword = tokenizer.decode([subwords[i]])
                if token not in subword_dict:
                    subword_dict[token] = subword
            except:
                pass

new_tokens = list(subword_dict.keys())
new_subwords = list(subword_dict.values())

unused_tokens = []
unmatched_tokens = []
tokens_to_assign = [] #unmatched_tokens not included in tokens_to_assign - separate treatment

for token in pretrained_tokens:
    if token not in new_tokens:
        unused_tokens.append(token) #get unused tokens in trained model
    else:
        #get tokens with same token number but different token subword
        if d_swap[token] !=subword_dict[token]:
            unmatched_tokens.append(token)
        else:
            pass

#get remaining new tokens we need to assign
for token in new_tokens:
    if token not in unmatched_tokens:
        tokens_to_assign.append(token)

#assign unused tokens new values
if len(unused_tokens) <= len(new_tokens):
    print('not enough tokens')
else:
    for i in range(len(unmatched_tokens)):
        d_swap[unused_tokens[i]] = subword_dict[unmatched_tokens[i]]

    for i in range(len(tokens_to_assign)):
        d_swap[tokens_to_assign[i]] = d_swap.pop(unused_tokens[i+len(unmatched_tokens)])
        d_swap[tokens_to_assign[i]] = subword_dict[tokens_to_assign[i]]

final_result = {v: k for k, v in d_swap.items()}
print(final_result)
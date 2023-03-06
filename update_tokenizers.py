from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import pandas as pd
import os
import json

'''
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
        if d_swap[token] != subword_dict[token]:
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

tokenizer = AutoTokenizer.from_pretrained("Davlan/afro-xlmr-mini")

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
    
    tokenized_data = dataset.map(lambda examples: self(examples[lang]), batched=True)
    # loop through the tokenized text data 
    for tokens in tokenized_data:
        subwords = tokens["input_ids"]
        # add the subwords for each token to the subword dictionary 
        for i in range(len(subwords)):
            token = subwords[i]
            try:
                subword = self.decode([subwords[i]])
                if token not in subword_dict:
                    subword_dict[token] = subword
            except:
                pass
    

with open('convert.txt', 'w') as convert_file:
     convert_file.write(dumps(subword_dict))
'''
'''
class UpdatedAutoTokenizer(AutoTokenizer):
    def __init__(self):
        super().__init__()
        self.pretrained_vocab = self.get_vocab()
        self.d_swap = {v: k for k, v in self.pretrained_vocab.items()}
        self.pretrained_tokens = list(self.pretrained_vocab.values())

    def update_tokenizer(self, dataset, split="train"):
        subword_dict = json.read(dataset)
        
        new_tokens = list(subword_dict.keys())

        unused_tokens = []
        unmatched_tokens = []
        tokens_to_assign = [] #unmatched_tokens not included in tokens_to_assign - separate treatment

        for token in self.pretrained_tokens:
            if token not in new_tokens:
                unused_tokens.append(token) #get unused tokens in trained model
            else:
                #get tokens with same token number but different token subword
                if self.d_swap[token] != subword_dict[token]:
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
                self.d_swap[unused_tokens[i]] = subword_dict[unmatched_tokens[i]]

            for i in range(len(tokens_to_assign)):
                self.d_swap[tokens_to_assign[i]] = self.d_swap.pop(unused_tokens[i+len(unmatched_tokens)])
                self.d_swap[tokens_to_assign[i]] = subword_dict[tokens_to_assign[i]]
        return {v: k for k, v in self.d_swap.items()}


model_name = "Davlan/afro-xlmr-mini"
tokenizer = UpdatedAutoTokenizer()
dataset = "Sunbird/salt-dataset"
split = "train"

tokenizer.update_tokenizer('convert.txt', split)
'''
'''
    def decode(self, input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        decoded_text = []
        for token in input_ids:
            if token in self.subword_dict:
                decoded_text.append(self.subword_dict[token])
            else:
                decoded_text.append(token)
        return decoded_text

tokenizer = AutoTokenizer.from_pretrained("Davlan/afro-xlmr-mini")

dataset = load_dataset("Sunbird/salt-dataset", split="train")
languages = dataset.column_names
subword_dict = {}
tokenized_data = {}
for lang in languages:
    tokenized_data = dataset.map(lambda examples: tokenizer(examples[lang]), batched=True)
    for tokens in tokenized_data:
        subwords = tokens["input_ids"]
        for i in range(len(subwords)):
            token = subwords[i]
            try:
                subword = tokenizer.decode([subwords[i]])
                if token not in subword_dict:
                    subword_dict[token] = subword
            except:
                pass

updated_tokenizer = UpdatedAutoTokenizer("Davlan/afro-xlmr-mini", subword_dict=subword_dict)
print(updated_tokenizer.final_result)


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
        if d_swap[token] != subword_dict[token]:
            unmatched_tokens.append(token)
        else:
            pass

#get remaining new tokens we need to assign
for token in new_tokens:
    if token not in unmatched_tokens:
        tokens_to_assign.append(token)

def add_language_tokens(tokenizer, tokens_to_replace, language_tokens):
    for old, new in zip(tokens_to_replace, language_tokens):
        if (">>" + new + "<<") not in tokenizer.encoder and old in tokenizer.encoder:
            tokenizer.encoder[(">>" + new + "<<")] = tokenizer.encoder[old]
            del tokenizer.encoder[old]
    return tokenizer

add_language_tokens(tokenizer, unused_tokens, tokens_to_assign)
'''
'''
class UpdatedAutoTokenizer(AutoTokenizer):
    def __init__(self, model_name, *args, **kwargs):
        super().__init__(model_name, *args, **kwargs)
        self.pretrained_vocab = self.get_vocab()
        self.d_swap = {v: k for k, v in self.pretrained_vocab.items()}
        self.pretrained_tokens = list(self.pretrained_vocab.values())

    def update_tokenize(self, dataset, split):
        subword_dict = {}
        for example in dataset[split]:
            for word in example['sentence']:
                subwords = self.tokenize(word)
                for subword in subwords:
                    if subword not in self.pretrained_tokens:
                        if subword not in subword_dict:
                            subword_dict[subword] = 1
                        else:
                            subword_dict[subword] += 1

        for subword, count in subword_dict.items():
            idx = len(self.pretrained_tokens) + len(subword_dict) - subword_dict[subword]
            self.pretrained_vocab[subword] = idx
            self.d_swap[idx] = subword
            self.pretrained_tokens.append(subword)

        with open(f"{self.name_or_path}-subword_dict.json", "w") as f:
            json.dump(subword_dict, f)

        new_tokens = list(subword_dict.keys())
        new_subwords = list(subword_dict.values())

        unused_tokens = []
        unmatched_tokens = []
        tokens_to_assign = []  # unmatched_tokens not included in tokens_to_assign - separate treatment

        for token in self.pretrained_tokens:
            if token not in new_tokens:
                unused_tokens.append(token)  # get unused tokens in trained model
            else:
                # get tokens with same token number but different token subword
                if self.d_swap[token] != subword_dict[token]:
                    unmatched_tokens.append(token)
                else:
                    pass

        # get remaining new tokens we need to assign
        for token in new_tokens:
            if token not in unmatched_tokens:
                tokens_to_assign.append(token)

        # assign unused tokens new values
        if len(unused_tokens) <= len(new_tokens):
            print("not enough tokens")
        else:
            for i in range(len(unmatched_tokens)):
                self.d_swap[unused_tokens[i]] = subword_dict[unmatched_tokens[i]]

            for i in range(len(tokens_to_assign)):
                self.d_swap[tokens_to_assign[i]] = self.d_swap.pop(unused_tokens[i + len(unmatched_tokens)])


dataset = load_dataset("Sunbird/salt-dataset")

tokenizer = AutoTokenizer.from_pretrained("Davlan/afro-xlmr-mini")

# Create an instance of UpdatedAutoTokenizer
updated_tokenizer = UpdatedAutoTokenizer(tokenizer)

# Now you can call update_tokenize method on the updated_tokenizer object
final_result = updated_tokenizer.update_tokenize(dataset, split='train')

print(final_result)

encoded = tokenizer.encode("Hello World")
print(encoded)

from transformers import AutoTokenizer

class UpdatedAutoTokenizer(AutoTokenizer):
    def __init__(self, tokenizer):
        super().__init__(tokenizer.model_name_or_path)
        self.tokenizer = tokenizer
        self.added_tokens = []

    def add_tokens(self, new_tokens):
        num_added_tokens = self.tokenizer.add_tokens(new_tokens)
        self.added_tokens.extend(new_tokens)
        return num_added_tokens

    def update_tokenize(self, dataset, split="train", max_length=512):
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], truncation=True, padding="max_length", max_length=max_length
            )

        return dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=dataset.column_names, keep_in_memory=True)

    def save_pretrained(self, *args, **kwargs):
        self.tokenizer.save_pretrained(*args, **kwargs)
        
    def get_vocab_size(self):
        return len(self.tokenizer)

from transformers import AutoTokenizer
from update_tokenizers import UpdatedAutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Davlan/afro-xlmr-mini")
updated_tokenizer = UpdatedAutoTokenizer(tokenizer)
'''
from transformers import PreTrainedTokenizerFast

class CustomTokenizer(PreTrainedTokenizerFast):
    def __init__(self):
        super().__init__(
            vocab_file=None, 
            tokenizer_file=None,
            model_max_length=None
        )

    def update_tokenizer(self, new_tokens):
        # Add new tokens to the vocabulary
        self.add_tokens(new_tokens, special_tokens=False)
        
    def tokenize_and_encode(self, text, **kwargs):
        # Tokenize and encode text using the updated tokenizer
        # You can define your own custom encoding logic here
        return self.encode(text, **kwargs)

tokenizer = CustomTokenizer()
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.add_tokens(['<new_token_1>', '<new_token_2>'], special_tokens=False)
tokenizer.save_pretrained('<path_to_save_updated_tokenizer>')

#####TRY THESE TWO

import torch

class Tokenizer:
    def __init__(self, name_or_path):
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(name_or_path)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.max_seq_length = self.tokenizer.model_max_length

        with open(f"{name_or_path}-vocab.json", "r") as f:
            self.pretrained_vocab = json.load(f)
        with open(f"{name_or_path}-d_swap.json", "r") as f:
            self.d_swap = json.load(f)

        self.pretrained_tokens = list(self.pretrained_vocab.keys())

    def tokenize(self, sentence):
        return self.tokenizer.tokenize(sentence)

    def encode(self, sentence):
        encoded_dict = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return encoded_dict['input_ids'], encoded_dict['attention_mask']

    def tokenize_and_encode(self, dataset):
        subword_dict = {}
        for example in dataset:
            for word in example['sentence']:
                subwords = self.tokenize(word)
                for subword in subwords:
                    if subword not in self.pretrained_tokens:
                        if subword not in subword_dict:
                            subword_dict[subword] = 1
                        else:
                            subword_dict[subword] += 1

        for subword, count in subword_dict.items():
            idx = len(self.pretrained_tokens) + len(subword_dict) - subword_dict[subword]
            self.pretrained_vocab[subword] = idx
            self.d_swap[idx] = subword
            self.pretrained_tokens.append(subword)

        with open(f"{self.name_or_path}-subword_dict.json", "w") as f:
            json.dump(subword_dict, f)

        new_tokens = list(subword_dict.keys())
        new_subwords = list(subword_dict.values())

        unused_tokens = []
        unmatched_tokens = []
        tokens_to_assign = []  # unmatched_tokens not included in tokens_to_assign - separate treatment

        for token in self.pretrained_tokens:
            if token not in new_tokens:
                unused_tokens.append(token)  # get unused tokens in trained model
            else:
                # get tokens with same token number but different token subword
                if self.d_swap[token] != subword_dict[token]:
                    unmatched_tokens.append(token)
                else:
                    pass

        # get remaining new tokens we need to assign
        for token in new_tokens:
            if token not in unmatched_tokens:
                tokens_to_assign.append(token)

        # assign unused tokens new values
        if len(unused_tokens) <= len(new_tokens):
            print("not enough tokens")
        else:
            for i in range(len(unmatched_tokens)):
                self.d_swap[unused_tokens[i]] = subword_dict[unmatched_tokens[i]]

            for i in range(len(tokens_to_assign)):
                self.d_swap[tokens_to_assign[i]] = self.d_swap.pop(unused_tokens[i + len(unmatched_tokens)])

        input_ids = []
        attention_masks = []

        for example in dataset:
            tokens = []
            for word in example['sentence']:
                subwords = self.tokenize(word)
                for subword in subwords:
                    if subword in self.pretrained_tokens:
                       

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import json
import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from datasets import load_from_disk

Data = load_from_disk("SALT_SPLIT")
# SALT_SPLIT contains English, src and src_lang

# Lug, lgg, ach, teo, nyn, swa

dictionary_list = []
sentences = []
labels = []
fields = ['sentence', 'English', 'Luganda', 'Runyankole', 'Ateso', 'Lugbara', 'Acholi','Swahili']

lgg = Data.filter(lambda ex: ex["src_lang"] == "lgg")
lug = Data.filter(lambda ex: ex["src_lang"] == "lug")
ach = Data.filter(lambda ex: ex["src_lang"] == "ach")
teo = Data.filter(lambda ex: ex["src_lang"] == "teo")
swa = Data.filter(lambda ex: ex["src_lang"] == "swa")
nyn = Data.filter(lambda ex: ex["src_lang"] == "nyn")

lug_length = len(lug["train"])
nyn_length = len(nyn["train"])
teo_length = len(teo["train"])
lgg_length = len(lgg["train"])
ach_length = len(ach["train"])

#print(lug_length,nyn_length,teo_length,lgg_length,ach_length)

# Add SALT training examples to dictionary list

for i in range(lug_length):
    new_dict = {'English': 1, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': lug["train"][i]['English']}
    dictionary_list.append(new_dict)
    new_dict2 = {'English': 0, 'Luganda': 1, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': lug["train"][i]['src']}
    dictionary_list.append(new_dict2)

for i in range(nyn_length):
    new_dict = {'English': 1, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': nyn["train"][i]['English']}
    dictionary_list.append(new_dict)
    new_dict2 = {'English': 0, 'Luganda': 0, 'Runyankole': 1, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': nyn["train"][i]['src']}
    dictionary_list.append(new_dict2)

for i in range(teo_length):
    new_dict = {'English': 1, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': teo["train"][i]['English']}
    dictionary_list.append(new_dict)
    new_dict2 = {'English': 0, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 1, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': teo["train"][i]['src']}
    dictionary_list.append(new_dict2)

for i in range(lgg_length):
    new_dict = {'English': 1, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': lgg["train"][i]['English']}
    dictionary_list.append(new_dict)
    new_dict2 = {'English': 0, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 1, 'Acholi':0, 'Swahili':0,'sentence': lgg["train"][i]['src']}
    dictionary_list.append(new_dict2)

for i in range(ach_length):
    new_dict = {'English': 1, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': ach["train"][i]['English']}
    dictionary_list.append(new_dict)
    new_dict2 = {'English': 0, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':1, 'Swahili':0,'sentence': ach["train"][i]['src']}
    dictionary_list.append(new_dict2)

print(len(dictionary_list))

sentences_and_labels = [[] for x in range(len(dictionary_list))]

counter = 0
for item in dictionary_list:
    sentences.append(item['sentence'])
    sentences_and_labels[counter].append(item['sentence'])
    labels.append([item['English'], item['Luganda'], item['Runyankole'],
                   item['Ateso'], item['Lugbara'], item['Acholi'], item['Swahili']])
    sentences_and_labels[counter].append(item['English'])
    sentences_and_labels[counter].append(item['Luganda'])
    sentences_and_labels[counter].append(item['Runyankole'])
    sentences_and_labels[counter].append(item['Ateso'])
    sentences_and_labels[counter].append(item['Lugbara'])
    sentences_and_labels[counter].append(item['Acholi'])
    sentences_and_labels[counter].append(item['Swahili'])
    counter += 1

print('sentences and labels length',len(sentences_and_labels))

# Create csv file to store training set
with open("sunbirdData_train.csv", "w", encoding="utf-8",newline="") as f:
    writer = csv.writer(f)
    writer.writerow(fields)
    writer.writerows(sentences_and_labels)

df = pd.read_csv("sunbirdData.csv", encoding='cp1252')
print(df.shape)
print(df["sentence"][168])


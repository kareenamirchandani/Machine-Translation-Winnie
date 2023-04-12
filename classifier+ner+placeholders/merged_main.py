# This file is a prototype that puts together the classifier and NER, together with the placeholders

from ner_func import ner_func
from predict_language import predict_language

import string
import numpy as np
import re


testsentence = "hey how are you Ttivvi nnyingi   UGANDA Peko ka  uganda CHristian university awanu Ms patricia M Egwelete itunga iboro are nnnyingi nuikamunitos ekuse ka etepe nnyingi ka u in France's patricia m with anna?"

# Step 1: Identify the matrix language

matrix_lang, perc_matrix_lang = predict_language(testsentence) 
#print(matrix_lang)


# Step 2: NER and NE placeholders

with open('NER/ner_dataset.txt','r', encoding='utf-8') as f:  # Open the NER dataset
  ner_dataset = set(f.read().split('\n'))
  ner_dataset = ner_dataset.difference({' ',''})

ne_dict = ner_func(testsentence, ner_dataset)  # ner_func() takes as input the sentence as a string and the NER dataset as a list and 
# returns a dictionary of the form {named entity: corresponding NE placeholder}

# NE placeholders:
testsentence = testsentence.lower()
#print(testsentence)
for ne in ne_dict:
  testsentence = re.sub(re.escape(ne), ne_dict[ne], testsentence)
print(testsentence) 



# Step 3: Identify fragments in embedded languages and replace them by placeholders

emb_dict = {}  # Dictionary of the form: {string in embedded language: list of the form [language as string, placeholder as string]}
emb_array = np.asarray([])
emb_array_lang = np.asarray([])
prev_emb_lan = ''
emb_index = 0


for word in testsentence.split():
    if '[NE' not in word:  # Don't consider named entities; cannot use ner_dict.values() because we want to account for e.g. "[NE3]'s"
        current_emb_lang, perc_emb_lang=predict_language(word)
        print(word)
        print(current_emb_lang)

        if current_emb_lang != matrix_lang:
            if current_emb_lang != prev_emb_lang:
                prev_fragment = word
                emb_array = np.append(emb_array,word)
                emb_array_lang = np.append(emb_array_lang, current_emb_lang)
                emb_index+=1
            else:
                emb_array[emb_index-1] = prev_fragment+' '+word
                prev_fragment = prev_fragment+word
            prev_emb_lang = current_emb_lang
        else:
            prev_emb_lang = ''
        
    else:
        prev_emb_lang = ''


emb_array, indices_unique = np.unique(emb_array, return_index=True)
emb_array_lang = emb_array_lang[indices_unique]

for i in range(len(emb_array)):
   emb_dict[emb_array[i]] = [emb_array_lang[i],f'[EMB{i+1}]']

print(ne_dict)
print(emb_dict)

# EMB placeholders:
import re

emb_list = list(emb_array)
emb_list.sort(key=len, reverse=True) # Sort in descending order of string length to avoid replacing e.g. 'ka' before 'nnyingi ka'
print(emb_array)
print(emb_list)
for emb in emb_list:   
  testsentence = re.sub(re.escape(' '+emb.lower()+' '), ' '+emb_dict[emb][1]+' ', testsentence)  # check lower cases?
print(testsentence) 


# Step 4: Translation


# Step 5: Final translated sentence 
# To translate embedded language fragments use emb_dict = {string in embedded language: list of the form [language as string, placeholder as string]}
# Assume testsentence below is the translated sentence with placeholders inside
# To go back from NE placeholders to NE:
for ne in ne_dict:
  testsentence = re.sub(re.escape(ne_dict[ne]), string.capwords(ne), testsentence)  # capwords used to capitalize the first letter of each word in the named entities
print(testsentence)

# To go from EMB placeholders to translated sections:
# e.g. create a dictionary transl_emb_dict = {EMB placeholder: translated fragment}
#for emb in transl_emb_dict:
    #testsentence = re.sub(re.escape(emb),transl_emb_dict[emb], testsentence)


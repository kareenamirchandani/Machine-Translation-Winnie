# This script is used to create the dataset used for NER list check method.
# (run on Google Colab due to issues with loading datasets on certain OS versions)


# The following 4 NER datasets were used:
# Dataset 1: chenyuxuan/wikigold -general NER dataset sourced from Wikipedia
# Dataset 2: wnut_17 -unusual and rare entities
# Dataset 3: masakhaner, lug -entities specififc to Luganda
# Dataset 4: masakhaner, swa -entities specific to Swahili

# Compound words and case-insensitive words were taken into account and punctuation was removed


from datasets import load_dataset
import numpy as np
import string


# Dataset 1: chenyuxuan/wikigold
# Process the dataset:
d1=load_dataset("chenyuxuan/wikigold")  # On some OS versions this doesn't work because wikigold.py doesn't have encoding='utf-8' specified; works in colab

array_chenyuxuanwikigold_complete=np.asarray([])

for elem in d1['train']:
  i=0
  while (i<len(elem['ner_tags'])):
    if elem['ner_tags'][i]:
      str_ne=elem['tokens'][i]
      j=i+1
      while j<len(elem['ner_tags']) and elem['ner_tags'][j]==elem['ner_tags'][i]:  # Account for compound words (depends on how the initial dataset is made)
        str_ne=str_ne+' '+elem['tokens'][j]
        j+=1
      i=j
      array_chenyuxuanwikigold_complete=np.append(array_chenyuxuanwikigold_complete,str_ne.translate(str.maketrans('','',string.punctuation)).lower())
    else:
      i+=1


# Dataset 2: wnut_17
# Process the dataset:
d2=load_dataset("wnut_17")

array_wnut17_complete=np.asarray([])

for elem in d2['train']:
  i=0
  while (i<len(elem['ner_tags'])):
    if elem['ner_tags'][i]:
      str_ne=elem['tokens'][i]
      j=i+1
      while j<len(elem['ner_tags']) and elem['ner_tags'][j]==elem['ner_tags'][i]+1:  # Account for compound words (depends on how the initial dataset is made)
        str_ne=str_ne+' '+elem['tokens'][j]
        j+=1
      i=j
      array_wnut17_complete=np.append(array_wnut17_complete,str_ne.translate(str.maketrans('','',string.punctuation)).lower())
    else:
      i+=1



# Dataset 3: masakhaner, lug
# Process the dataset:
d3=load_dataset("masakhaner",'lug')

array_masakhaner_lug_complete=np.asarray([])

for elem in d3['train']:
  i=0
  while (i<len(elem['ner_tags'])):
    if elem['ner_tags'][i]:
      str_ne=elem['tokens'][i]
      j=i+1
      while j<len(elem['ner_tags']) and elem['ner_tags'][j]==elem['ner_tags'][i]+1:  # Account for compound words (depends on how the initial dataset is made)
        str_ne=str_ne+' '+elem['tokens'][j]
        j+=1
      i=j
      array_masakhaner_lug_complete=np.append(array_masakhaner_lug_complete,str_ne.translate(str.maketrans('','',string.punctuation)).lower())
    else:
      i+=1



# Dataset 3: masakhaner, lug
# Process the dataset:
d4=load_dataset("masakhaner","swa")

array_masakhaner_swa_complete=np.asarray([])

for elem in d4['train']:
  i=0
  while (i<len(elem['ner_tags'])):
    if elem['ner_tags'][i]:
      str_ne=elem['tokens'][i]
      j=i+1
      while j<len(elem['ner_tags']) and elem['ner_tags'][j]==elem['ner_tags'][i]+1:  # Account for compound words (depends on how the initial dataset is made)
        str_ne=str_ne+' '+elem['tokens'][j]
        j+=1
      i=j
      array_masakhaner_swa_complete=np.append(array_masakhaner_swa_complete,str_ne.translate(str.maketrans('','',string.punctuation)).lower())
    else:
      i+=1

# Concatenate all datasets to form the final NER dataset:
ner_array_complete=np.concatenate((array_chenyuxuanwikigold_complete, array_wnut17_complete, array_masakhaner_lug_complete, array_masakhaner_swa_complete))

# To remove duplicates:  
ner_array_complete, indices_unique=np.unique(ner_array_complete, return_index=True)



# Save ner_array_complete in a txt file ner_dataset in Drive:

# Mount driver to authenticate yourself to gdrive
from google.colab import drive
drive.mount('/content/gdrive')

# Import necessary libraries
from numpy import savetxt

# save to txt file
savetxt('ner_dataset.txt', ner_array_complete, fmt='%s')

# To load the dataset in the form of a list of named entities:
with open('ner_dataset.txt','r') as f:
  ner_dataset=f.read().split()


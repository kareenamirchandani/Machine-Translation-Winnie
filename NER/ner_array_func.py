# Given an input sentence and the path to a NER dataset, the function return the NE from the dataset contained in the sentence


# Function for NER general use:

def ner_array_func(testsentence,ner_dataset_path):
  import string
  import re

  with open(ner_dataset_path,'r', encoding='utf-8') as f:  # Open the NER dataset
      ner_dataset = set(f.read().split('\n'))
      ner_dataset = list(ner_dataset.difference({' ',''}))

  ne_array_list=[]

  testsentence1=testsentence  # Can modify the copy without modifying the original because strings are immutable in Python

  ner_dataset=list(ner_dataset)

  ner_dataset.sort(key=len, reverse=True) # Sort in descending order of string length to avoid replacing e.g. simple words like 'Uganda' before compound entities like 'Uganda Christian University'

  testsentence1= testsentence1.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))  # to replace punctuation by spaces
  testsentence1=testsentence1.lower()  # Convert the test sentence to lower case to make the NER case-insensitive

  for item in ner_dataset:
    if testsentence1.find(" "+item+" ")!=-1:
      ne_array_list.append(string.capwords(item))
      testsentence1= testsentence1.translate(str.maketrans('','',item))

  return ne_array_list


# For testing accuracy: (only open the dataset once, before calling the function)

def ner_array_func_test(testsentence,ner_dataset):
  import string
  import re

  ne_array_list=[]

  testsentence1=testsentence  # Can modify the copy without modifying the original because strings are immutable in Python

  testsentence1= testsentence1.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))) # to replace punctuation by spaces
  testsentence1=testsentence1.lower()  # Convert the test sentence to lower case to make the NER case-insensitive

  for item in ner_dataset:
    if testsentence1.find(" "+item+" ")!=-1:
      ne_array_list.append(string.capwords(item))
      testsentence1= testsentence1.translate(str.maketrans('','',item))

  return ne_array_list

# Function for NER:

def ner_func(testsentence, ner_dataset):
  import string
  import re

  ne_dict={}
  ne_index=1

  testsentence1=testsentence  # Can modify the copy without modifying the original because strings are immutable in Python

  ner_dataset=list(ner_dataset)

  ner_dataset.sort(key=len, reverse=True) # Sort in descending order of string length to avoid replacing e.g. simple words like 'Uganda' before compound entities like 'Uganda Christian University'

  testsentence1= testsentence1.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))  # to replace punctuation by spaces
  testsentence1=testsentence1.lower()  # Convert the test sentence to lower case to make the NER case-insensitive

  for item in ner_dataset:
    if testsentence1.find(" "+item+" ")!=-1:
      ne_dict[item]=f'[NE{ne_index}]'
      testsentence1 = re.sub(re.escape(' '+item+' '), ' ', testsentence1)
      ne_index+=1

  return ne_dict


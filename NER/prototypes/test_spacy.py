import string
import re
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

from ner_masakhaner_func import ner_masakhaner_func_test
from ner_array_func import ner_array_func_test
from ner_spacy_func import ner_spacy_english_func_test, ner_spacy_multilingual_func_test
from common_words_func import common_words_func_test


testsentence1 = "ho Uganda is focusing on farming."
testsentence2 = "hey Uganda essira eritadde ku bulimi."

# For ner_array_func_test:
with open('ner_dataset.txt','r', encoding='utf-8') as f:  # Open the NER dataset
    ner_dataset = set(f.read().split('\n'))
ner_dataset = list(ner_dataset.difference({' ',''}))
ner_dataset.sort(key=len, reverse=True) # Sort in descending order of string length to avoid replacing e.g. simple words like 'Uganda' before compound entities like 'Uganda Christian University'


# For ner_spacy_english_func_test and ner_spacy_multilingual_func_test:
nlp_english = spacy.load("en_core_web_sm")
nlp_multilingual = spacy.load('xx_ent_wiki_sm')

# For ner_masakhaner_func_test:
tokenizer = AutoTokenizer.from_pretrained("masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0")
model = AutoModelForTokenClassification.from_pretrained("masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0")
nlp_masakhaner = pipeline("ner", model=model, tokenizer=tokenizer) 




my_list = ner_spacy_english_func_test(testsentence1, nlp_english)
spmul = ner_spacy_multilingual_func_test(testsentence1, nlp_multilingual)
comm = common_words_func_test(testsentence1,testsentence2)
mas= ner_masakhaner_func_test(testsentence1,nlp_masakhaner)
arr = ner_array_func_test(testsentence1, ner_dataset)


print(my_list)
print(spmul)
print(comm)
print(mas)
print(arr)

print(type(my_list[0]))
print(type(comm[0]))
print(my_list[0]== comm[0])

print(set(comm).intersection(set(my_list), set(spmul),set(mas),set(arr)))
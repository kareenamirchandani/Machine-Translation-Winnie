# This script is used to assess the 3 different methods of NER: simple list check, SpaCy, AfriXLMR-large finetuned on MasakhaNER 1.0 and 2.0

import string
import re
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

from ner_masakhaner_func import ner_masakhaner_func_test
from ner_array_func import ner_array_func_test
from ner_spacy_func import ner_spacy_english_func_test, ner_spacy_multilingual_func_test
from common_words_func import common_words_func_test  # Point of reference to test the accuracy of the NER methods

from datasets import load_dataset

salt_dataset = load_dataset('Sunbird/salt-dataset')  # SALT dataset is used to test english, luganda, lugbara, acholi, ateso and runyankole
salt_eng = salt_dataset['train']['English']
salt_lgd = salt_dataset['train']['Luganda']
salt_lgb = salt_dataset['train']['Lugbara']
salt_ach = salt_dataset['train']['Acholi']
salt_ats = salt_dataset['train']['Ateso']
salt_rnk = salt_dataset['train']['Runyankole']

with open('mt560_swa_dataset.txt','r', encoding='utf-8') as f:  # MT560 is used to test swahili
    mt560_swa = list(f.read().split('\n'))

with open('mt560_eng_dataset.txt','r', encoding='utf-8') as f:  # MT560 is used to test swahili
    mt560_eng = list(f.read().split('\n'))

# To assess the NER array method:
false_positives_eng_array = 0
false_positives_lgd_array = 0
false_positives_lgb_array = 0
false_positives_ach_array = 0
false_positives_ats_array = 0
false_positives_rnk_array = 0
false_positives_swa_array = 0

false_negatives_eng_array = 0
false_negatives_lgd_array = 0
false_negatives_lgb_array = 0
false_negatives_ach_array = 0
false_negatives_ats_array = 0
false_negatives_rnk_array = 0
false_negatives_swa_array = 0

true_positives_eng_array = 0
true_positives_lgd_array = 0
true_positives_lgb_array = 0
true_positives_ach_array = 0
true_positives_ats_array = 0
true_positives_rnk_array = 0
true_positives_swa_array = 0

true_negatives_eng_array = 0
true_negatives_lgd_array = 0
true_negatives_lgb_array = 0
true_negatives_ach_array = 0
true_negatives_ats_array = 0
true_negatives_rnk_array = 0
true_negatives_swa_array = 0

# To assess the NER using the english spacy model:
false_positives_eng_spacy_en = 0
false_positives_lgd_spacy_en = 0
false_positives_lgb_spacy_en = 0
false_positives_ach_spacy_en = 0
false_positives_ats_spacy_en = 0
false_positives_rnk_spacy_en = 0
false_positives_swa_spacy_en = 0

false_negatives_eng_spacy_en = 0
false_negatives_lgd_spacy_en = 0
false_negatives_lgb_spacy_en = 0
false_negatives_ach_spacy_en = 0
false_negatives_ats_spacy_en = 0
false_negatives_rnk_spacy_en = 0
false_negatives_swa_spacy_en = 0

true_positives_eng_spacy_en = 0
true_positives_lgd_spacy_en = 0
true_positives_lgb_spacy_en = 0
true_positives_ach_spacy_en = 0
true_positives_ats_spacy_en = 0
true_positives_rnk_spacy_en = 0
true_positives_swa_spacy_en = 0

true_negatives_eng_spacy_en = 0
true_negatives_lgd_spacy_en = 0
true_negatives_lgb_spacy_en = 0
true_negatives_ach_spacy_en = 0
true_negatives_ats_spacy_en = 0
true_negatives_rnk_spacy_en = 0
true_negatives_swa_spacy_en = 0

# To assess the NER using the multilingual spacy model:
false_positives_eng_spacy_mul = 0
false_positives_lgd_spacy_mul = 0
false_positives_lgb_spacy_mul = 0
false_positives_ach_spacy_mul = 0
false_positives_ats_spacy_mul = 0
false_positives_rnk_spacy_mul = 0
false_positives_swa_spacy_mul = 0

false_negatives_eng_spacy_mul = 0
false_negatives_lgd_spacy_mul = 0
false_negatives_lgb_spacy_mul = 0
false_negatives_ach_spacy_mul = 0
false_negatives_ats_spacy_mul = 0
false_negatives_rnk_spacy_mul = 0
false_negatives_swa_spacy_mul = 0

true_positives_eng_spacy_mul = 0
true_positives_lgd_spacy_mul = 0
true_positives_lgb_spacy_mul = 0
true_positives_ach_spacy_mul = 0
true_positives_ats_spacy_mul = 0
true_positives_rnk_spacy_mul = 0
true_positives_swa_spacy_mul = 0

true_negatives_eng_spacy_mul = 0
true_negatives_lgd_spacy_mul = 0
true_negatives_lgb_spacy_mul = 0
true_negatives_ach_spacy_mul = 0
true_negatives_ats_spacy_mul = 0
true_negatives_rnk_spacy_mul = 0
true_negatives_swa_spacy_mul = 0

# To assess the NER using the masakhaner model:
false_positives_eng_masakhaner = 0
false_positives_lgd_masakhaner = 0
false_positives_lgb_masakhaner = 0
false_positives_ach_masakhaner = 0
false_positives_ats_masakhaner = 0
false_positives_rnk_masakhaner = 0
false_positives_swa_masakhaner = 0

false_negatives_eng_masakhaner = 0
false_negatives_lgd_masakhaner = 0
false_negatives_lgb_masakhaner = 0
false_negatives_ach_masakhaner = 0
false_negatives_ats_masakhaner = 0
false_negatives_rnk_masakhaner = 0
false_negatives_swa_masakhaner = 0

true_positives_eng_masakhaner = 0
true_positives_lgd_masakhaner = 0
true_positives_lgb_masakhaner = 0
true_positives_ach_masakhaner = 0
true_positives_ats_masakhaner = 0
true_positives_rnk_masakhaner = 0
true_positives_swa_masakhaner = 0

true_negatives_eng_masakhaner = 0
true_negatives_lgd_masakhaner = 0
true_negatives_lgb_masakhaner = 0
true_negatives_ach_masakhaner = 0
true_negatives_ats_masakhaner = 0
true_negatives_rnk_masakhaner = 0
true_negatives_swa_masakhaner = 0


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


#for i in range(len(salt_dataset['train']['English']): 
for i in range (25000):
    
    # English:
    common_words_list_eng = common_words_func_test(salt_eng[i],salt_lgd[i]) 
    ne_array_list_eng = ner_array_func_test(' '+salt_eng[i]+' ',ner_dataset)
    ne_spacy_en_list_eng = ner_spacy_english_func_test(salt_eng[i],nlp_english)
    ne_spacy_mul_list_eng = ner_spacy_multilingual_func_test(salt_eng[i], nlp_multilingual)
    ne_masakhaner_list_eng = ner_masakhaner_func_test(salt_eng[i], nlp_masakhaner)

    true_positives_eng_array += len(set(common_words_list_eng).intersection(set(ne_array_list_eng)))
    true_positives_eng_spacy_en += len(set(common_words_list_eng).intersection(set(ne_spacy_en_list_eng)))
    true_positives_eng_spacy_mul += len(set(common_words_list_eng).intersection(set(ne_spacy_mul_list_eng)))
    true_positives_eng_masakhaner += len(set(common_words_list_eng).intersection(set(ne_masakhaner_list_eng)))

    true_negatives_eng_array += len((set(salt_eng[i].split()).difference(set(common_words_list_eng))).difference(set(ne_array_list_eng)))
    true_negatives_eng_spacy_en += len((set(salt_eng[i].split()).difference(set(common_words_list_eng))).difference(set(ne_spacy_en_list_eng)))
    true_negatives_eng_spacy_mul += len((set(salt_eng[i].split()).difference(set(common_words_list_eng))).difference(set(ne_spacy_mul_list_eng)))
    true_negatives_eng_masakhaner += len((set(salt_eng[i].split()).difference(set(common_words_list_eng))).difference(set(ne_masakhaner_list_eng)))

    false_positives_eng_array += len(set(ne_array_list_eng).difference(set(common_words_list_eng).intersection(set(ne_array_list_eng))))
    false_positives_eng_spacy_en += len(set(ne_spacy_en_list_eng).difference(set(common_words_list_eng).intersection(set(ne_spacy_en_list_eng))))
    false_positives_eng_spacy_mul += len(set(ne_spacy_mul_list_eng).difference(set(common_words_list_eng).intersection(set(ne_spacy_mul_list_eng))))
    false_positives_eng_masakhaner += len(set(ne_masakhaner_list_eng).difference(set(common_words_list_eng).intersection(set(ne_masakhaner_list_eng))))

    false_negatives_eng_array += len(set(common_words_list_eng).difference(set(common_words_list_eng).intersection(set(ne_array_list_eng))))
    false_negatives_eng_spacy_en += len(set(common_words_list_eng).difference(set(common_words_list_eng).intersection(set(ne_spacy_en_list_eng))))
    false_negatives_eng_spacy_mul += len(set(common_words_list_eng).difference(set(common_words_list_eng).intersection(set(ne_spacy_mul_list_eng))))
    false_negatives_eng_masakhaner += len(set(common_words_list_eng).difference(set(common_words_list_eng).intersection(set(ne_masakhaner_list_eng))))

    # Luganda:
    common_words_list_lgd = common_words_func_test(salt_eng[i],salt_lgd[i]) # Use the english-source language pair to assess the accuracy of ner methods on source language
    ne_array_list_lgd = ner_array_func_test(' '+salt_lgd[i]+' ',ner_dataset)
    ne_spacy_en_list_lgd = ner_spacy_english_func_test(salt_lgd[i], nlp_english)
    ne_spacy_mul_list_lgd = ner_spacy_multilingual_func_test(salt_lgd[i], nlp_multilingual)
    ne_masakhaner_list_lgd = ner_masakhaner_func_test(salt_lgd[i], nlp_masakhaner)

    true_positives_lgd_array += len(set(common_words_list_lgd).intersection(set(ne_array_list_lgd)))
    true_positives_lgd_spacy_en += len(set(common_words_list_lgd).intersection(set(ne_spacy_en_list_lgd)))
    true_positives_lgd_spacy_mul += len(set(common_words_list_lgd).intersection(set(ne_spacy_mul_list_lgd)))
    true_positives_lgd_masakhaner += len(set(common_words_list_lgd).intersection(set(ne_masakhaner_list_lgd)))

    true_negatives_lgd_array += len((set(salt_lgd[i].split()).difference(set(common_words_list_lgd))).difference(set(ne_array_list_lgd)))
    true_negatives_lgd_spacy_en += len((set(salt_lgd[i].split()).difference(set(common_words_list_lgd))).difference(set(ne_spacy_en_list_lgd)))
    true_negatives_lgd_spacy_mul += len((set(salt_lgd[i].split()).difference(set(common_words_list_lgd))).difference(set(ne_spacy_mul_list_lgd)))
    true_negatives_lgd_masakhaner += len((set(salt_lgd[i].split()).difference(set(common_words_list_lgd))).difference(set(ne_masakhaner_list_lgd)))


    false_positives_lgd_array += len(set(ne_array_list_lgd).difference(set(common_words_list_lgd).intersection(set(ne_array_list_lgd))))
    false_positives_lgd_spacy_en += len(set(ne_spacy_en_list_lgd).difference(set(common_words_list_lgd).intersection(set(ne_spacy_en_list_lgd))))
    false_positives_lgd_spacy_mul += len(set(ne_spacy_mul_list_lgd).difference(set(common_words_list_lgd).intersection(set(ne_spacy_mul_list_lgd))))
    false_positives_lgd_masakhaner += len(set(ne_masakhaner_list_lgd).difference(set(common_words_list_lgd).intersection(set(ne_masakhaner_list_lgd))))

    false_negatives_lgd_array += len(set(common_words_list_lgd).difference(set(common_words_list_lgd).intersection(set(ne_array_list_lgd))))
    false_negatives_lgd_spacy_en += len(set(common_words_list_lgd).difference(set(common_words_list_lgd).intersection(set(ne_spacy_en_list_lgd))))
    false_negatives_lgd_spacy_mul += len(set(common_words_list_lgd).difference(set(common_words_list_lgd).intersection(set(ne_spacy_mul_list_lgd))))
    false_negatives_lgd_masakhaner += len(set(common_words_list_lgd).difference(set(common_words_list_lgd).intersection(set(ne_masakhaner_list_lgd))))

    # Lugbara:
    common_words_list_lgb = common_words_func_test(salt_eng[i],salt_lgb[i]) 
    ne_array_list_lgb = ner_array_func_test(' '+salt_lgb[i]+' ',ner_dataset)
    ne_spacy_en_list_lgb = ner_spacy_english_func_test(salt_lgb[i], nlp_english)
    ne_spacy_mul_list_lgb = ner_spacy_multilingual_func_test(salt_lgb[i], nlp_multilingual)
    ne_masakhaner_list_lgb = ner_masakhaner_func_test(salt_lgb[i], nlp_masakhaner)

    true_positives_lgb_array += len(set(common_words_list_lgb).intersection(set(ne_array_list_lgb)))
    true_positives_lgb_spacy_en += len(set(common_words_list_lgb).intersection(set(ne_spacy_en_list_lgb)))
    true_positives_lgb_spacy_mul += len(set(common_words_list_lgb).intersection(set(ne_spacy_mul_list_lgb)))
    true_positives_lgb_masakhaner += len(set(common_words_list_lgb).intersection(set(ne_masakhaner_list_lgb)))

    true_negatives_lgb_array += len((set(salt_lgb[i].split()).difference(set(common_words_list_lgb))).difference(set(ne_array_list_lgb)))
    true_negatives_lgb_spacy_en += len((set(salt_lgb[i].split()).difference(set(common_words_list_lgb))).difference(set(ne_spacy_en_list_lgb)))
    true_negatives_lgb_spacy_mul += len((set(salt_lgb[i].split()).difference(set(common_words_list_lgb))).difference(set(ne_spacy_mul_list_lgb)))
    true_negatives_lgb_masakhaner += len((set(salt_lgb[i].split()).difference(set(common_words_list_lgb))).difference(set(ne_masakhaner_list_lgb)))

    false_positives_lgb_array += len(set(ne_array_list_lgb).difference(set(common_words_list_lgb).intersection(set(ne_array_list_lgb))))
    false_positives_lgb_spacy_en += len(set(ne_spacy_en_list_lgb).difference(set(common_words_list_lgb).intersection(set(ne_spacy_en_list_lgb))))
    false_positives_lgb_spacy_mul += len(set(ne_spacy_mul_list_lgb).difference(set(common_words_list_lgb).intersection(set(ne_spacy_mul_list_lgb))))
    false_positives_lgb_masakhaner += len(set(ne_masakhaner_list_lgb).difference(set(common_words_list_lgb).intersection(set(ne_masakhaner_list_lgb))))

    false_negatives_lgb_array += len(set(common_words_list_lgb).difference(set(common_words_list_lgb).intersection(set(ne_array_list_lgb))))
    false_negatives_lgb_spacy_en += len(set(common_words_list_lgb).difference(set(common_words_list_lgb).intersection(set(ne_spacy_en_list_lgb))))
    false_negatives_lgb_spacy_mul += len(set(common_words_list_lgb).difference(set(common_words_list_lgb).intersection(set(ne_spacy_mul_list_lgb))))
    false_negatives_lgb_masakhaner += len(set(common_words_list_lgb).difference(set(common_words_list_lgb).intersection(set(ne_masakhaner_list_lgb))))

    
    # Acholi:
    common_words_list_ach = common_words_func_test(salt_eng[i],salt_ach[i]) 
    ne_array_list_ach = ner_array_func_test(' '+salt_ach[i]+' ',ner_dataset)
    ne_spacy_en_list_ach = ner_spacy_english_func_test(salt_ach[i], nlp_english)
    ne_spacy_mul_list_ach = ner_spacy_multilingual_func_test(salt_ach[i], nlp_multilingual)
    ne_masakhaner_list_ach = ner_masakhaner_func_test(salt_ach[i], nlp_masakhaner)

    true_positives_ach_array += len(set(common_words_list_ach).intersection(set(ne_array_list_ach)))
    true_positives_ach_spacy_en += len(set(common_words_list_ach).intersection(set(ne_spacy_en_list_ach)))
    true_positives_ach_spacy_mul += len(set(common_words_list_ach).intersection(set(ne_spacy_mul_list_ach)))
    true_positives_ach_masakhaner += len(set(common_words_list_ach).intersection(set(ne_masakhaner_list_ach)))

    true_negatives_ach_array += len((set(salt_ach[i].split()).difference(set(common_words_list_ach))).difference(set(ne_array_list_ach)))
    true_negatives_ach_spacy_en += len((set(salt_ach[i].split()).difference(set(common_words_list_ach))).difference(set(ne_spacy_en_list_ach)))
    true_negatives_ach_spacy_mul += len((set(salt_ach[i].split()).difference(set(common_words_list_ach))).difference(set(ne_spacy_mul_list_ach)))
    true_negatives_ach_masakhaner += len((set(salt_ach[i].split()).difference(set(common_words_list_ach))).difference(set(ne_masakhaner_list_ach)))

    false_positives_ach_array += len(set(ne_array_list_ach).difference(set(common_words_list_ach).intersection(set(ne_array_list_ach))))
    false_positives_ach_spacy_en += len(set(ne_spacy_en_list_ach).difference(set(common_words_list_ach).intersection(set(ne_spacy_en_list_ach))))
    false_positives_ach_spacy_mul += len(set(ne_spacy_mul_list_ach).difference(set(common_words_list_ach).intersection(set(ne_spacy_mul_list_ach))))
    false_positives_ach_masakhaner += len(set(ne_masakhaner_list_ach).difference(set(common_words_list_ach).intersection(set(ne_masakhaner_list_ach))))

    false_negatives_ach_array += len(set(common_words_list_ach).difference(set(common_words_list_ach).intersection(set(ne_array_list_ach))))
    false_negatives_ach_spacy_en += len(set(common_words_list_ach).difference(set(common_words_list_ach).intersection(set(ne_spacy_en_list_ach))))
    false_negatives_ach_spacy_mul += len(set(common_words_list_ach).difference(set(common_words_list_ach).intersection(set(ne_spacy_mul_list_ach))))
    false_negatives_ach_masakhaner += len(set(common_words_list_ach).difference(set(common_words_list_ach).intersection(set(ne_masakhaner_list_ach))))

    # Ateso:
    common_words_list_ats = common_words_func_test(salt_eng[i],salt_ats[i]) 
    ne_array_list_ats = ner_array_func_test(' '+salt_ats[i]+' ',ner_dataset)
    ne_spacy_en_list_ats = ner_spacy_english_func_test(salt_ats[i], nlp_english)
    ne_spacy_mul_list_ats = ner_spacy_multilingual_func_test(salt_ats[i], nlp_multilingual)
    ne_masakhaner_list_ats = ner_masakhaner_func_test(salt_ats[i], nlp_masakhaner)

    true_positives_ats_array += len(set(common_words_list_ats).intersection(set(ne_array_list_ats)))
    true_positives_ats_spacy_en += len(set(common_words_list_ats).intersection(set(ne_spacy_en_list_ats)))
    true_positives_ats_spacy_mul += len(set(common_words_list_ats).intersection(set(ne_spacy_mul_list_ats)))
    true_positives_ats_masakhaner += len(set(common_words_list_ats).intersection(set(ne_masakhaner_list_ats)))

    true_negatives_ats_array += len((set(salt_ats[i].split()).difference(set(common_words_list_ats))).difference(set(ne_array_list_ats)))
    true_negatives_ats_spacy_en += len((set(salt_ats[i].split()).difference(set(common_words_list_ats))).difference(set(ne_spacy_en_list_ats)))
    true_negatives_ats_spacy_mul += len((set(salt_ats[i].split()).difference(set(common_words_list_ats))).difference(set(ne_spacy_mul_list_ats)))
    true_negatives_ats_masakhaner += len((set(salt_ats[i].split()).difference(set(common_words_list_ats))).difference(set(ne_masakhaner_list_ats)))

    false_positives_ats_array += len(set(ne_array_list_ats).difference(set(common_words_list_ats).intersection(set(ne_array_list_ats))))
    false_positives_ats_spacy_en += len(set(ne_spacy_en_list_ats).difference(set(common_words_list_ats).intersection(set(ne_spacy_en_list_ats))))
    false_positives_ats_spacy_mul += len(set(ne_spacy_mul_list_ats).difference(set(common_words_list_ats).intersection(set(ne_spacy_mul_list_ats))))
    false_positives_ats_masakhaner += len(set(ne_masakhaner_list_ats).difference(set(common_words_list_ats).intersection(set(ne_masakhaner_list_ats))))

    false_negatives_ats_array += len(set(common_words_list_ats).difference(set(common_words_list_ats).intersection(set(ne_array_list_ats))))
    false_negatives_ats_spacy_en += len(set(common_words_list_ats).difference(set(common_words_list_ats).intersection(set(ne_spacy_en_list_ats))))
    false_negatives_ats_spacy_mul += len(set(common_words_list_ats).difference(set(common_words_list_ats).intersection(set(ne_spacy_mul_list_ats))))
    false_negatives_ats_masakhaner += len(set(common_words_list_ats).difference(set(common_words_list_ats).intersection(set(ne_masakhaner_list_ats))))

    # Runyankole:
    common_words_list_rnk = common_words_func_test(salt_eng[i],salt_rnk[i]) 
    ne_array_list_rnk = ner_array_func_test(' '+salt_rnk[i]+' ',ner_dataset)
    ne_spacy_en_list_rnk = ner_spacy_english_func_test(salt_rnk[i], nlp_english)
    ne_spacy_mul_list_rnk = ner_spacy_multilingual_func_test(salt_rnk[i], nlp_multilingual)
    ne_masakhaner_list_rnk = ner_masakhaner_func_test(salt_rnk[i], nlp_masakhaner)

    true_positives_rnk_array += len(set(common_words_list_rnk).intersection(set(ne_array_list_rnk)))
    true_positives_rnk_spacy_en += len(set(common_words_list_rnk).intersection(set(ne_spacy_en_list_rnk)))
    true_positives_rnk_spacy_mul += len(set(common_words_list_rnk).intersection(set(ne_spacy_mul_list_rnk)))
    true_positives_rnk_masakhaner += len(set(common_words_list_rnk).intersection(set(ne_masakhaner_list_rnk)))

    true_negatives_rnk_array += len((set(salt_rnk[i].split()).difference(set(common_words_list_rnk))).difference(set(ne_array_list_rnk)))
    true_negatives_rnk_spacy_en += len((set(salt_rnk[i].split()).difference(set(common_words_list_rnk))).difference(set(ne_spacy_en_list_rnk)))
    true_negatives_rnk_spacy_mul += len((set(salt_rnk[i].split()).difference(set(common_words_list_rnk))).difference(set(ne_spacy_mul_list_rnk)))
    true_negatives_rnk_masakhaner += len((set(salt_rnk[i].split()).difference(set(common_words_list_rnk))).difference(set(ne_masakhaner_list_rnk)))

    false_positives_rnk_array += len(set(ne_array_list_rnk).difference(set(common_words_list_rnk).intersection(set(ne_array_list_rnk))))
    false_positives_rnk_spacy_en += len(set(ne_spacy_en_list_rnk).difference(set(common_words_list_rnk).intersection(set(ne_spacy_en_list_rnk))))
    false_positives_rnk_spacy_mul += len(set(ne_spacy_mul_list_rnk).difference(set(common_words_list_rnk).intersection(set(ne_spacy_mul_list_rnk))))
    false_positives_rnk_masakhaner += len(set(ne_masakhaner_list_rnk).difference(set(common_words_list_rnk).intersection(set(ne_masakhaner_list_rnk))))

    false_negatives_rnk_array += len(set(common_words_list_rnk).difference(set(common_words_list_rnk).intersection(set(ne_array_list_rnk))))
    false_negatives_rnk_spacy_en += len(set(common_words_list_rnk).difference(set(common_words_list_rnk).intersection(set(ne_spacy_en_list_rnk))))
    false_negatives_rnk_spacy_mul += len(set(common_words_list_rnk).difference(set(common_words_list_rnk).intersection(set(ne_spacy_mul_list_rnk))))
    false_negatives_rnk_masakhaner += len(set(common_words_list_rnk).difference(set(common_words_list_rnk).intersection(set(ne_masakhaner_list_rnk))))


    # Swahili:
    common_words_list_swa = common_words_func_test(mt560_eng[i],mt560_swa[i]) # Use the english-source language pair to assess the accuracy of ner methods on source language
    ne_array_list_swa = ner_array_func_test(' '+mt560_swa[i]+' ',ner_dataset)
    ne_spacy_en_list_swa = ner_spacy_english_func_test(mt560_swa[i], nlp_english)
    ne_spacy_mul_list_swa = ner_spacy_multilingual_func_test(mt560_swa[i], nlp_multilingual)
    ne_masakhaner_list_swa = ner_masakhaner_func_test(mt560_swa[i], nlp_masakhaner)

    true_positives_swa_array += len(set(common_words_list_swa).intersection(set(ne_array_list_swa)))
    true_positives_swa_spacy_en += len(set(common_words_list_swa).intersection(set(ne_spacy_en_list_swa)))
    true_positives_swa_spacy_mul += len(set(common_words_list_swa).intersection(set(ne_spacy_mul_list_swa)))
    true_positives_swa_masakhaner += len(set(common_words_list_swa).intersection(set(ne_masakhaner_list_swa)))

    true_negatives_swa_array += len((set(mt560_swa[i].split()).difference(set(common_words_list_swa))).difference(set(ne_array_list_swa)))
    true_negatives_swa_spacy_en += len((set(mt560_swa[i].split()).difference(set(common_words_list_swa))).difference(set(ne_spacy_en_list_swa)))
    true_negatives_swa_spacy_mul += len((set(mt560_swa[i].split()).difference(set(common_words_list_swa))).difference(set(ne_spacy_mul_list_swa)))
    true_negatives_swa_masakhaner += len((set(mt560_swa[i].split()).difference(set(common_words_list_swa))).difference(set(ne_masakhaner_list_swa)))

    false_positives_swa_array += len(set(ne_array_list_swa).difference(set(common_words_list_swa).intersection(set(ne_array_list_swa))))
    false_positives_swa_spacy_en += len(set(ne_spacy_en_list_swa).difference(set(common_words_list_swa).intersection(set(ne_spacy_en_list_swa))))
    false_positives_swa_spacy_mul += len(set(ne_spacy_mul_list_swa).difference(set(common_words_list_swa).intersection(set(ne_spacy_mul_list_swa))))
    false_positives_swa_masakhaner += len(set(ne_masakhaner_list_swa).difference(set(common_words_list_swa).intersection(set(ne_masakhaner_list_swa))))

    false_negatives_swa_array += len(set(common_words_list_swa).difference(set(common_words_list_swa).intersection(set(ne_array_list_swa))))
    false_negatives_swa_spacy_en += len(set(common_words_list_swa).difference(set(common_words_list_swa).intersection(set(ne_spacy_en_list_swa))))
    false_negatives_swa_spacy_mul += len(set(common_words_list_swa).difference(set(common_words_list_swa).intersection(set(ne_spacy_mul_list_swa))))
    false_negatives_swa_masakhaner += len(set(common_words_list_swa).difference(set(common_words_list_swa).intersection(set(ne_masakhaner_list_swa))))


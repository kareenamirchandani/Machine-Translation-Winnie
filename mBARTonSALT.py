# Finetuning mBART on SALT

from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from datasets import load_dataset
import torch

# get mBART model and tokenizer
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", tgt_lang = "en_XX", src_lang = "te_IN")
# Going to reuse language code for for ugandan languages, target is always going to be english
# src_lang code not to be used as special tokens are ignored and added manually

# Get SALT dataset
dataset = load_dataset("Sunbird/salt-dataset", split="train")
# Languages and the codes we will use. Attempted to use language codes that are underrepresented in pre-training data set
languages = ["Luganda", "Runyankole", "Ateso", "Lugbara", "Acholi"]
unusedLangCodes = ["te_IN", "si_LK", "bn_IN", "ml_IN", "ne_NP"]
langToCodes = dict(zip(languages, unusedLangCodes))
oldToNewCode = {"te_IN" : "lu_UG",
                "si_LK" : "ru_UG",
                "bn_IN" : "at_UG",
                "ml_IN" : "lg_UG",
                "ne_NP" : "ac_UG"}
newToOldCode = dict((v,k) for k,v in oldToNewCode.items())

def preprocess(example):
    inputs = []
    targets = []

    for i in range(0,len(example["English"])):
        targets = targets + [example["English"][i] for lang in languages]
        inputs = inputs + [langToCodes[lang] + example[lang][i] + tokenizer.eos_token for lang in languages ]

    model_inputs = tokenizer(inputs, add_special_tokens = False,  max_length = 128, truncation = True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length = 128, truncation = True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

model_inputs = preprocess(dataset)
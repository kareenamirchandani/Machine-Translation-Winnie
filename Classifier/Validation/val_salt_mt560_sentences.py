# This file is for outputting predictions of validation sentences
# The order is English, Luganda, Runyankole, Ateso, Lugbara, Acholi

from keras.models import load_model
import numpy as np
import pickle
import csv
from keras.utils import pad_sequences
from datasets import load_from_disk
import itertools

max_length = 100
trunc_type = 'post'
padding_type = 'post'
Data = load_from_disk("../SALT_SPLIT")
MT560 = load_from_disk('C:/Users/emike/MT560')


lgg = Data.filter(lambda ex: ex["src_lang"] == "lgg")
lug = Data.filter(lambda ex: ex["src_lang"] == "lug")
ach = Data.filter(lambda ex: ex["src_lang"] == "ach")
teo = Data.filter(lambda ex: ex["src_lang"] == "teo")
swa = Data.filter(lambda ex: ex["src_lang"] == "swa")
nyn = Data.filter(lambda ex: ex["src_lang"] == "nyn")

lugMT560 = MT560.filter(lambda ex: ex["src_lang"] == "lug")
swaMT560 = MT560.filter(lambda ex: ex["src_lang"] == "swa")
achMT560 = MT560.filter(lambda ex: ex["src_lang"] == "ach")
nynMT560 = MT560.filter(lambda ex: ex["src_lang"] == "nyn")

lug_length = len(lug["test"])
nyn_length = len(nyn["test"])
teo_length = len(teo["test"])
lgg_length = len(lgg["test"])
ach_length = len(ach["test"])

lug_lengthMT560 = len(lugMT560["test"])
swa_lengthMT560 = len(swaMT560["test"])
ach_lengthMT560 = len(achMT560["test"])
nyn_lengthMT560 = len(nynMT560["test"])

eng_list = []
lug_list = []
nyn_list = []
teo_list = []
lgg_list = []
ach_list = []
swa_list = []

for i in range(lug_length):
    eng_list.append(lug["test"][i]['English'])
    lug_list.append(lug["test"][i]['src'])

for i in range(nyn_length):
    eng_list.append(nyn["test"][i]['English'])
    nyn_list.append(nyn["test"][i]['src'])

for i in range(teo_length):
    eng_list.append(teo["test"][i]['English'])
    teo_list.append(teo["test"][i]['src'])

for i in range(lgg_length):
    eng_list.append(lgg["test"][i]['English'])
    lgg_list.append(lgg["test"][i]['src'])

for i in range(ach_length):
    eng_list.append(ach["test"][i]['English'])
    ach_list.append(ach["test"][i]['src'])

for i in range(swa_lengthMT560):
    eng_list.append(swaMT560["test"][i]['English'])
    swa_list.append(swaMT560["test"][i]['src'])

for i in range(ach_lengthMT560):
    eng_list.append(achMT560["test"][i]['English'])
    ach_list.append(achMT560["test"][i]['src'])

for i in range(lug_lengthMT560):
    eng_list.append(lugMT560["test"][i]['English'])
    lug_list.append(lugMT560["test"][i]['src'])

for i in range(nyn_lengthMT560):
    eng_list.append(nynMT560["test"][i]['English'])
    nyn_list.append(nynMT560["test"][i]['src'])

# Load the tokenizer from the base model
with open('../tokenizer4.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the finetuned model

#model = load_model('../trained_model_softmax.h5')
#model = load_model('../finetuned_softmax_word.h5')
#model = load_model('../lang_classifier_softmax2.h5')
#model = load_model('../lang_classifier_softmax4.h5')
model = load_model('../SALT_MT560_finetuned_word.h5')
model.summary()

print(len(swa_list))
validation_sentence = swa_list[0:5000]


val_sequences = tokenizer.texts_to_sequences(validation_sentence)
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

proba = model.predict(val_padded)
proba = np.asarray(proba, dtype=float)

predictions = model.predict(val_padded)
predictions_max = predictions.argmax(axis=1)

'''
for i,item in enumerate(predictions):
    if item.argmax(axis=0) == 0 and item[0] < 0.5:
        predictions_max[i] = predictions[i][1:].argmax(axis=0)'''

eng_count = np.count_nonzero(predictions_max == 0)
lug_count = np.count_nonzero(predictions_max == 1)
nyn_count = np.count_nonzero(predictions_max == 2)
teo_count = np.count_nonzero(predictions_max == 3)
lgg_count = np.count_nonzero(predictions_max == 4)
ach_count = np.count_nonzero(predictions_max == 5)
swa_count = np.count_nonzero(predictions_max == 6)

language_counts = [eng_count, lug_count, nyn_count, teo_count, lgg_count, ach_count, swa_count]

print('shape',np.shape(predictions_max))
print('Language counts', language_counts)



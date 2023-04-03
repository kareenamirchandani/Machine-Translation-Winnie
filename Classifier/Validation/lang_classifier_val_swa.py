# This file is for outputting predictions of validation sentences
# The order is English, Luganda, Runyankole, Ateso, Lugbara, Acholi

from keras.models import load_model
import numpy as np
import pickle

from keras.utils import pad_sequences

max_length = 100
trunc_type = 'post'
padding_type = 'post'

# Load the tokenizer from the base model
with open('../tokenizer4.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the finetuned model
#model = load_model('../finetuned_softmax_word.h5')
#model = load_model('../lang_classifier_softmax2.h5')
model = load_model('../lang_classifier_softmax4.h5')
model.summary()

swa_list_val = []

with open('C:/Users/emike/dev/swh.txt', encoding='utf-8') as f:
    swa_lines = f.readlines()

for item in swa_lines:
    swa_list_val.append(item.strip())

validation_sentence = swa_list_val

val_sequences = tokenizer.texts_to_sequences(validation_sentence)
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

proba = model.predict(val_padded)
proba = np.asarray(proba, dtype=float)

predictions = model.predict(val_padded)
predictions_max = predictions.argmax(axis=1)

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
print(sum(language_counts))


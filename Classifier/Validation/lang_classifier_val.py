# This file is for outputting predictions of validation sentences
# The order is English, Luganda, Runyankole, Ateso, Lugbara, Acholi

from keras.models import load_model
import numpy as np
import pickle

from keras.utils import pad_sequences

vocab_size = 100000  # tokenizer will keep the top 50000 words
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

# Load the tokenizer from the base model
with open('../tokenizer_softmax.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the finetuned model
model = load_model('../finetuned_softmax_word.h5')
#model = load_model('../lang_classifier_softmax2.h5')
model.summary()


# Luganda validation sentence
# Translates to 'Could I use your phone?'

validation_sentence = ['land']
#validation_sentence = ['ombuzireho omukundwa wangye']
#validation_sentence = ['serpents in the trees ombuzireho']
#validation_sentence = ['ombuzireho omukundwa wangye especially in the afternoon']
#validation_sentence = ['ombuzireho omukundwa wangye'] # I miss you so much darling (Runyankole)
#validation_sentence = ['Arai ilosi ijo, abuni aupar ke ijo.'] # Translates to 'If you are going, I shall accompany you'
#validation_sentence = ['I will go to work as soon as the sun rises in the sky.']
#validation_sentence = ['Mwattu yogera mpolampola'] # Please speak more slowly (in Luganda)
#validation_sentence = ['I will not go to work todat']
#validation_sentence = ['land']
#validation_sentence = ['Hello, what is your name?']
#validation_sentence = ['How much land does that farmer own.']
#validation_sentence = ['Nsabaku simu yo']
val_sequences = tokenizer.texts_to_sequences(validation_sentence)
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

#for x in range(len(val_padded)):
#    print(val_padded[x], 'length:',len(val_padded[x]))
#print('val_padded',val_padded)
#print('val_padded',len(val_padded))
proba = model.predict(val_padded)

#print('len of proba',len(proba))
# print(proba)
proba = np.asarray(proba, dtype=float)

# Calculate the sum (this should be equal to 1)
sum = np.sum(proba, 1)

print(' Prediction:', proba, 'sum:',sum)

# Format the output by rounding proba to 5dp and outputting percentage
counter = 0
languages = ['English', 'Luganda', 'Runyankole', 'Ateso', 'Lugbara', 'Acholi', 'Swahili']
for item in proba[0]:
    print(languages[counter] + ': ' + str((round(item,5)*100))+'%')
    counter += 1

from keras.models import load_model
import numpy as np
import pickle

from keras.utils import pad_sequences

vocab_size = 90000  # tokenizer will keep the top 50000 words
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 105025

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
model = load_model('trained_model_softmax.h5')
model.summary()


# Luganda validation sentence
# Translates to 'Could I use your phone?'

#validation_sentence = ['Morning']
validation_sentence = ['ombuzireho omukundwa wangye']
#validation_sentence = ['ombuzireho omukundwa wangye especially in the afternoon']
#validation_sentence = ['ombuzireho omukundwa wangye'] # I miss you so much darling (Runyankole)
#validation_sentence = ['Arai ilosi ijo, abuni aupar ke ijo.'] # Translates to 'If you are going, I shall accompany you'
#validation_sentence = ['I will go to work as soon as the sun rises in the sky.']
#validation_sentence = ['Mwattu yogera mpolampola', 'I will not go to work todat'] # Please speak more slowly (in Luganda)
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

sum = np.sum(proba, 1)

print(' Prediction:', proba, 'sum:',sum)

'''
# validation_sentence = 'Arai ilosi ijo, abuni aupar ke ijo.' Ateso
# Translates to 'If you are going, I shall accompany you'


# English validation sentence

validation_sentence = 'The sun is shining, it is a beautiful day.'
val_sequences = tokenizer.texts_to_sequences(validation_sentence)
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

proba = model.predict(val_padded)
proba = np.asarray(proba, dtype=float)

prediction = np.sum(proba, 0)

print('English prediction:', prediction)

# The predictions aren't really making sense
# Look for more multilabel text examples
'''
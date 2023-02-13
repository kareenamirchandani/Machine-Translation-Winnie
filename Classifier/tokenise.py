import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer

# Following the NLP Zero to Hero zeries on the Tensorflow yt channel

# Documentation on the tokenizer class
# https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/text.py#L138
sentences = [
    'I love my dog dog dog dog',
    'Bbiringanya lubeerera  asinga kukulira mu mbeera ya bugumu'
]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)  # Creates vocab based on word frequency
word_index = tokenizer.word_index  # Each word gets unique integer
print(word_index)  # Lower integer means more frequent word
# 0 is reserved for padding

sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)

# You will fit on your training corpus once,
# and use that exact same word_index dictionary at,
# train / eval / testing / prediction time to convert actual,
# text into sequences to feed them to the network


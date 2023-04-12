# This file is for outputting predictions of validation sentences
# Given an input text, the function will output the most probable language and its associated percentage
# The order is English, Luganda, Runyankole, Ateso, Lugbara, Acholi

from keras.models import load_model
import numpy as np
import pickle

from keras.utils import pad_sequences

def predict_language(input):

    vocab_size = 100000  # tokenizer will keep the top 50000 words
    embedding_dim = 16
    max_length = 100
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"

    # Load the tokenizer from the base model
    with open('tokenizer4.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Load the finetuned model
    model = load_model('lang_classifier_softmax4.h5')


    validation_sentence = [input]
    val_sequences = tokenizer.texts_to_sequences(validation_sentence)
    val_padded = pad_sequences(val_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    proba = model.predict(val_padded)

    proba = np.asarray(proba, dtype=float)

    # Calculate the sum (this should be equal to 1)
    sum = np.sum(proba, 1)

    #print(' Prediction:', proba, 'sum:',sum)

    # Format the output by rounding proba to 5dp and outputting percentage
    counter = 0
    languages = ['English', 'Luganda', 'Runyankole', 'Ateso', 'Lugbara', 'Acholi', 'Swahili']
    '''for item in proba[0]:
        print(languages[counter] + ': ' + str((round(item,5)*100))+'%')
        counter += 1'''

    #output = '{}: {}'.format(languages[proba.argmax()], str((round(proba[0][proba.argmax()],5)*100))+'%')
    language=languages[proba.argmax()]
    percentage = str((round(proba[0][proba.argmax()],5)*100))+'%'

    #print('{}: {}'.format(languages[proba.argmax()], str((round(proba[0][proba.argmax()],5)*100))+'%'))

    #return output
    return language,percentage

language,percentage = predict_language('awanu')
print(language)
print(percentage)

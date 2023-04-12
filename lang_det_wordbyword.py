import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer


import csv

with open('sunbirdData_new.csv', encoding='UTF8') as csvfile:
    reader = csv.DictReader(csvfile)
    
    corpus_eng = []
    corpus_ach = []
    corpus_lgg = []
    corpus_lug = []
    corpus_run = []
    corpus_teo = []

    for row in reader:
        corpus_eng.append(row['English'])
        corpus_ach.append(row['Acholi'])
        corpus_lgg.append(row['Lugbara'])
        corpus_lug.append(row['Luganda'])
        corpus_run.append(row['Runyankole'])
        corpus_teo.append(row['Ateso'])




tokenizer_eng=Tokenizer(oov_token="<OOV>")
tokenizer_ach=Tokenizer(oov_token="<OOV>")
tokenizer_lgg=Tokenizer(oov_token="<OOV>")
tokenizer_lug=Tokenizer(oov_token="<OOV>")
tokenizer_run=Tokenizer(oov_token="<OOV>")
tokenizer_teo=Tokenizer(oov_token="<OOV>")


tokenizer_eng.fit_on_texts(corpus_eng)
tokenizer_ach.fit_on_texts(corpus_ach)
tokenizer_lgg.fit_on_texts(corpus_lgg)
tokenizer_lug.fit_on_texts(corpus_lug)
tokenizer_run.fit_on_texts(corpus_run)
tokenizer_teo.fit_on_texts(corpus_teo)

word_index_eng=tokenizer_eng.word_index
print(word_index_eng)


sent_to_tokenize=['hey how are redio ipu u?',' how are you feeling?']
sent_seq_eng=tokenizer_eng.texts_to_sequences(sent_to_tokenize)
sent_seq_ach=tokenizer_ach.texts_to_sequences(sent_to_tokenize)
sent_seq_lgg=tokenizer_lgg.texts_to_sequences(sent_to_tokenize)
sent_seq_lug=tokenizer_lug.texts_to_sequences(sent_to_tokenize)
sent_seq_run=tokenizer_run.texts_to_sequences(sent_to_tokenize)
sent_seq_teo=tokenizer_teo.texts_to_sequences(sent_to_tokenize)
print(sent_seq_eng)
print(sent_seq_ach)
print(sent_seq_lgg)
print(sent_seq_lug)
print(sent_seq_run)
print(sent_seq_teo)


#create a list containing all indices corresponding to a word given the different tokens from different languages
tokens_list=[]

for i in range (len(sent_seq_eng)):
  tokens_list.append([])
  for j in range (len(sent_seq_eng[i])):
    tokens_list[i].append([sent_seq_eng[i][j], sent_seq_ach[i][j], sent_seq_lgg[i][j], sent_seq_lug[i][j], sent_seq_run[i][j], sent_seq_teo[i][j]])

print(tokens_list)  #for each word we have a token index list with indexes in each language in the order [eng, ach, lgg, lug, run, teo]



#check what language each word comes from
#ideally most of the words will have index 1 (ie OOV) for all the languages except from one language which is the language from which the word comes
#special case 1: all indices for a word are 1, ie the word cannot be found in any of our corpus- check for mispelling and try to identify the language after- TO DO
#special case 2: more than two indices for one word are different than 1, ie the word is identified in multiple languages- possible solutions to pick a language:
#sol1 for special case 2: firstly check the words around and choose the a language if this is found in the word before or after (easy) or better translate from both languages and see which makes sense in context-TO DO
#sol2 for special case 2: if sol1 fails, choose the language we find the most in the text or do translate from both languages and see how it fits the context-TO DO
#if the word is from the embedded language then select it for translation

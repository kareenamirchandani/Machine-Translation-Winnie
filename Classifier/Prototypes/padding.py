from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

sentences = ['I have a pet dog', 'Do you have a pet cat?']
# Create tokenizer
tokenizer = Tokenizer(num_words=10, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
print(word_index)

validation_sentence = 'I love my pet'
val_sequences = tokenizer.texts_to_sequences(validation_sentence)
print(val_sequences)
val_padded = pad_sequences(val_sequences, maxlen=50, padding='post', truncating='post')

print(val_padded)
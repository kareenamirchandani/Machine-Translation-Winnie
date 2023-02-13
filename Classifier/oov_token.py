from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences)
# Padding is used to make all the sentences equally sized
# padded = pad_sequences(sequences, padding='post')
# Padding parameter can be used to put the zeroes at the end
# maxlen parameter can be used to specify max sen length
# truncating param used to specify 'pre' or 'post' truncation
test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]
print(word_index)
test_seq = tokenizer.texts_to_sequences(test_data)
print(test_seq)
# Using the oov token maintains the sequence length,
# even though some meaning is still lost

# This file is for outputting predictions of validation sentences
# The order is English, Luganda, Runyankole, Ateso, Lugbara, Acholi
from sklearn.metrics import confusion_matrix
from keras.models import load_model
import numpy as np
import pickle
import tensorflow as tf
from keras.utils import pad_sequences
import seaborn as sns
import matplotlib.pyplot as plt

vocab_size = 100000  # tokenizer will keep the top 50000 words
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

english_list_val = []

# Load the tokenizer from the base model
with open('../tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('C:/Users/emike/dev/eng.txt', encoding='utf-8') as f:
    eng_lines = f.readlines()

for item in eng_lines:
    english_list_val.append(item.strip())

# Load the finetuned model
#model = load_model('finetuned_softmax_word.h5')
model = load_model('../lang_classifier_softmax2.h5')
model.summary()


# Luganda validation sentence
# Translates to 'Could I use your phone?'


validation_sentence = english_list_val
val_sequences = tokenizer.texts_to_sequences(validation_sentence)
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


proba = model.predict(val_padded)
predictions = model.predict(val_padded)
print('max',predictions.argmax(axis=1))

eng_labels2 = []
for i in range(997):
    eng_labels2.append([0,0,0,0,0,1,0])

eng_labels2 = np.asarray(eng_labels2,dtype=float)
print('max labels',eng_labels2.argmax(axis=1))
#print(eng_labels2)
#print(len(eng_labels2.argmax(axis=1)))
#conf_matrix = tf.math.confusion_matrix(labels=eng_labels2,
#                                       predictions=predictions)

#confusion_matrix = confusion_matrix(eng_labels2, np.rint(predictions))
cm = confusion_matrix(y_true=eng_labels2.argmax(axis=1), y_pred= predictions.argmax(axis=1), labels= [0, 5])
print(cm)

ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='Greens');  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['English', 'Not English']); ax.yaxis.set_ticklabels(['English', 'Not English']);
plt.show()
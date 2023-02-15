import tensorflow as tf
import pandas as pd
import keras
import numpy as np
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from keras import models, layers, optimizers, losses, metrics
import pickle
from keras.utils import pad_sequences
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras import backend as K

vocab_size = 90000  # tokenizer will keep the top 50000 words
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 105025

df = pd.read_csv("sunbird_split_word.csv", encoding='ISO-8859-1')
df = df.replace(np.nan, '', regex=True)
#print(tabulate(df, headers='keys'))

df_labels = df[["English", "Luganda", "Runyankole", "Ateso", "Lugbara", "Acholi"]]

X = list(df["sentence"])
y = df_labels.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

base_model = tf.keras.models.load_model('trained_model_softmax.h5')

print("Number of layers in the base model: ", len(base_model.layers))

# Load tokenizer from base model
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

word_index = tokenizer.word_index

X_train = pad_sequences(X_train, padding=padding_type, maxlen=max_length, truncating=trunc_type)
X_test = pad_sequences(X_test, padding=padding_type, maxlen=max_length, truncating=trunc_type)

input_tensor = base_model.layers[1]

new_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    input_tensor,
    #tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    #tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
    #layers.Activation('sigmoid')(input_tensor)
])

#input_tensor = base_model.layers[1].output     # choose how many layers you want to keep
#h1 = layers.Dense(10, name='dense_new_1')(input_tensor)
#h2 = layers.Dense(1, name='dense_new_2')(h1)
#out = layers.Activation('softmax')(input_tensor)
#out = tf.keras.layers.Dense(6, activation='softmax')

#new_model = models.Model(base_model.input, outputs=out)

for i in range(len(base_model.layers)):
    layers.trainable = True   # True--> fine tine, False-->frozen

print("Number of layers in the new model: ", len(new_model.layers))

new_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

new_model.summary()

num_epochs = 8
history = new_model.fit(X_train, y_train, epochs=num_epochs,
                    validation_data=(X_test, y_test), verbose=2)


def plot_result(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


plot_result("loss")
plot_result("accuracy")

new_model.save("finetuned_softmax_word.h5")

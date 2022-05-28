import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

"""
!wget --no-check-certificate \
    https://raw.githubusercontent.com/aliakbarbadri/nlp-tf/master/irish-lyrics-eof.txt \
    -O /tmp/irish-lyrics-eof.txt
"""

# ---------------------------------- Tokenize the text
tokenizer = Tokenizer()

data = open('/tmp/irish-lyrics-eof.txt').read()

corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

print(tokenizer.word_index)
print(total_words)

# -------------------------- Convert text to ngrams sequence

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)
# input_sequences[:13]

# --------------------------------- Convert ngrams to same length sequences

# pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
print(input_sequences[:13])

# ---------------------------------  Get Xs and Ys (labels)

# create predictors and label
xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
print(xs[:13])
print("\n\n")
print(labels[:13])

# ----------------------------- Convert Ys to onehot

# one-hot
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
print(ys[:13])

# --------------------------------- NN
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len - 1))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words, activation='softmax'))
adam = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
# print(model.summary())
# print(model)
# --------------------------------------

# earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
history = model.fit(xs, ys, epochs=20, verbose=1)

# ------------------------------------- Prediction by model

seed_text = "I've got a bad feeling about this"
seed_text = "The true purpose of deep learning is making memes because"
next_words = 100
token_list = tokenizer.texts_to_sequences([seed_text])[0]

for _ in range(next_words):
    pedded_token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(pedded_token_list, verbose=0)
    classes_x = np.argmax(predicted, axis=1)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == classes_x:
            output_word = word
            break
    token_list.append(int(classes_x))
    seed_text += " " + output_word
print(seed_text)

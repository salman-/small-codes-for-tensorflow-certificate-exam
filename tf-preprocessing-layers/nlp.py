import zipfile
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization,Embedding,Input,GlobalAveragePooling1D,Dense,LSTM,GRU,Bidirectional,Conv1D,GlobalMaxPool1D
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub
import pandas as pd

# Download data (same as from Kaggle)
#!wget "https://storage.googleapis.com/ztm_tf_course/nlp_getting_started.zip"

# Unzip data
zip_ref = zipfile.ZipFile("nlp_getting_started.zip", "r")
zip_ref.extractall()
zip_ref.close()

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

def remove_stop_words(tweet):
  #print(tweet)
  stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
  tweet = tweet.lower()
  words = tweet.split(' ')
  non_stop_words = [w for w in words if w not in stopwords ]
  return (" ").join(non_stop_words)

train_df['text'] = train_df['text'].apply(lambda tweet : remove_stop_words(tweet) if tweet is not np.nan else tweet)

train_sentences,test_sentences,train_labels,target_labels = train_test_split(train_df['text'],train_df['target'],shuffle=True,test_size=0.2)

text_vectorization_layer = TextVectorization(max_tokens=10000,ngrams=15,output_mode='int',output_sequence_length=15)
text_vectorization_layer.adapt(train_sentences)

input = Input(shape=(1,),dtype='string')
x = text_vectorization_layer(input)
x = Embedding(input_dim=10,output_dim=128,embeddings_initializer='uniform')(x)
x = LSTM(64,return_sequences=True)(x)
x = LSTM(64)(x)

output = Dense(1,activation='sigmoid')(x)

model = tf.keras.models.Model(input,output)

model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

result= model.fit(x=train_sentences,y=train_labels,validation_data=(test_sentences,target_labels),epochs=6)

history = pd.DataFrame(result.history)
history[['accuracy','val_accuracy']].plot()

#Import

import sys
import pandas as pd
import tensorflow as tf
from termcolor import colored

#Python and Tensorflow version check

print(colored('Python Version: %s' % sys.version.split()[0], 'blue'))
print(colored('TensorFlow Ver: %s' % tf.__version__, 'magenta'))

#Feed forward + Backprop =epoch

n_epoch = int(input('Enter no. of epochs for RNN training: '))
print(colored('No. of epochs: %d' % n_epoch, 'red'))


pd.set_option('display.max_colwidth', 1000)

#Sqlite3 database

import pandas as pd
import sqlite3

#Connect to database

conn = sqlite3.connect('Twitter Database.db')
c = conn.cursor()

sent_ver = pd.read_sql_query('SELECT * FROM SentimentData',conn)

c.close()
conn.close()

#Tweet  

sent_ver.head(10)

#Number of tweet and shape

print('sent_ver.shape:', sent_ver.shape)

#Number of positive and negative data [1 = positive. 0 = negative]

sent_ver.Sentiment.value_counts()
sent_ver.head()

sent = sent_ver[ ['Sentiment','Tweets'] ].copy()
sent.head(10)

X = sent.Tweets
y = sent.Sentiment 

X.head(10)
y.head(10)

#Import tflearn

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences

#Train-Test-Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


#Create the vocab (so that we can create X_word_ids from X)

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(ngram_range=(1,1), token_pattern=r'\b\w{1,}\b')

vect.fit(X_train)
vocab = vect.vocabulary_

def convert_X_to_X_word_ids(X):
    return X.apply( lambda x: [vocab[w] for w in [w.lower().strip() for w in x.split()] if w in vocab] )


X_train_word_ids = convert_X_to_X_word_ids(X_train)
X_test_word_ids  = convert_X_to_X_word_ids(X_test)


#Difference between X(_train/_test) and X(_train_word_ids/test_word_ids)

X_train.head()

X_train_word_ids.head()

print('X_train_word_ids.shape:', X_train_word_ids.shape)
print('X_test_word_ids.shape:', X_test_word_ids.shape)

#Sequence Padding

X_train_padded_seqs = pad_sequences(X_train_word_ids, maxlen=20, value=0)
X_test_padded_seqs  = pad_sequences(X_test_word_ids , maxlen=20, value=0)

print('X_train_padded_seqs.shape:', X_train_padded_seqs.shape)
print('X_test_padded_seqs.shape:', X_test_padded_seqs.shape)


pd.DataFrame(X_train_padded_seqs).head()
pd.DataFrame(X_test_padded_seqs).head()

unique_y_labels = list(y_train.value_counts().index)
unique_y_labels

len(unique_y_labels)

#Preprocessing the data

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(unique_y_labels)

 print('')
 print(unique_y_labels)
 print(le.transform(unique_y_labels))
 print('')

 print('')
for label_id, label_name in zip(le.transform(unique_y_labels), unique_y_labels):
    print('%d: %s' % (label_id, label_name))
print('')    

y_train = to_categorical(y_train.map(lambda x: le.transform([x])[0]), nb_classes=len(unique_y_labels))
y_test  = to_categorical(y_test.map(lambda x:  le.transform([x])[0]), nb_classes=len(unique_y_labels))


y_train[0:3]


print('y_train.shape:', y_train.shape)
print('y_test.shape:', y_test.shape)

#Network Building

size_of_each_vector = X_train_padded_seqs.shape[1]
vocab_size = len(vocab)
no_of_unique_y_labels = len(unique_y_labels)

print('size_of_each_vector:', size_of_each_vector)
print('vocab_size:', vocab_size)
print('no_of_unique_y_labels:', no_of_unique_y_labels)

sgd = tflearn.SGD(learning_rate=1e-4, lr_decay=0.96, decay_step=1000)

net = tflearn.input_data([None, size_of_each_vector]) 
net = tflearn.embedding(net, input_dim=vocab_size, output_dim=128) 
net = tflearn.lstm(net, 128, dropout=0.6) 
net = tflearn.fully_connected(net, no_of_unique_y_labels, activation='softmax') 
net = tflearn.regression(net, 
                         optimizer='adam',
                         learning_rate=1e-4,
                         loss='categorical_crossentropy')


#Intialize the Model

model = tflearn.DNN(net, tensorboard_verbose=0, checkpoint_path='Model4/model.tfl.ckpt')
model = tflearn.DNN(net, tensorboard_verbose=0)


#Train

model.fit(X_train_padded_seqs, y_train, 
          validation_set=(X_test_padded_seqs, y_test), 
          n_epoch=n_epoch,
          show_metric=True, 
          batch_size=100)


#Manually Save the Model

model.save('Model1/model.tfl')
print(colored('Model Saved!', 'red'))

#Manually Load the Model

model.load('Model1/model.tfl')
print(colored('Model Loaded!', 'red'))



#RNN Accuracy

import numpy as np
from sklearn import metrics


pred_classes = [np.argmax(i) for i in model.predict(X_test_padded_seqs)]
true_classes = [np.argmax(i) for i in y_test]

print(colored('\nRNN Classifier\'s Accuracy: %0.5f\n' % metrics.accuracy_score(true_classes, pred_classes), 'cyan'))



#Show some predicted tweet

ids_of_titles = range(0,50) 

for i in ids_of_titles:
    pred_class = np.argmax(model.predict([X_test_padded_seqs[i]]))
    true_class = np.argmax(y_test[i])
    
    print(X_test.values[i])
    print('pred_class:', le.inverse_transform(pred_class))
    print('true_class:', le.inverse_transform(true_class))
    print('')


import numpy as np
import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, GlobalMaxPooling1D, LSTM
from keras.callbacks import ModelCheckpoint
from keras.models import Model

from dataload import load_obama


MAX_NB_WORDS = 10500
MAX_SEQUENCE_LENGTH = 140
VALIDATION_SPLIT = 0.1
EMBEDDING_DIM = 200

EMBEDDING_DIR = 'embedding'
EMBEDDING_TYPE = 'glove.twitter.27B.200d.txt' # 'glove.6B.%dd.txt' % (EMBEDDING_DIM)

texts, labels, label_map = load_obama()

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

embeddings_index = {}
f = open(os.path.join(EMBEDDING_DIR, EMBEDDING_TYPE), encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Preparing embedding matrix.')

# prepare embedding matrix
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


print('Training model.')

checkpoint = ModelCheckpoint('data/lstm_model (glove 300).h5', monitor='val_fbeta_score', verbose=2, save_weights_only=True,
                             save_best_only=True, mode='max')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = LSTM(256, dropout_W=0.2, dropout_U=0.2, return_sequences=False)(embedded_sequences)
#x = LSTM(256, dropout_W=0.2, dropout_U=0.2, return_sequences=False)(x)
# x = Conv1D(64, 5, activation='relu', border_mode='same')(embedded_sequences)
# x = MaxPooling1D(2)(x)
# x = Conv1D(128, 5, activation='relu', border_mode='same')(x)
# x = MaxPooling1D(2)(x)
# x = Conv1D(256, 5, activation='relu', border_mode='same')(x)
# x = MaxPooling1D(2)(x)
# x = Conv1D(256, 5, activation='relu', border_mode='same')(x)
# x = MaxPooling1D(2)(x)
# x = Conv1D(512, 5, activation='relu', border_mode='same')(x)
#x = GlobalMaxPooling1D()(x)
#x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
preds = Dense(3, activation='softmax')(x)

model = Model(sequence_input, preds)

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'fbeta_score'])

# happy learning!
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          callbacks=[checkpoint], nb_epoch=100, batch_size=100)


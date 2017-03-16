import numpy as np
import os

from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers import Embedding, LSTM
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras import backend as K

from keras_utils import load_both, load_embedding_matrix, prepare_tokenized_data, train_keras_model_cv


MAX_NB_WORDS = 16000
MAX_SEQUENCE_LENGTH = 140
VALIDATION_SPLIT = 0.1
EMBEDDING_DIM = 300

EMBEDDING_DIR = 'embedding'
EMBEDDING_TYPE = 'glove.6B.300d.txt' # 'glove.6B.%dd.txt' % (EMBEDDING_DIM)

texts, labels, label_map = load_both()

data, word_index = prepare_tokenized_data(texts, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)

# prepare embedding matrix
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = load_embedding_matrix(EMBEDDING_DIR + "/" + EMBEDDING_TYPE,
                                         word_index, MAX_NB_WORDS, EMBEDDING_DIM)

def gen_model():
    channel_axis = 1 if K.image_dim_ordering() == 'th' else -1
    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False, mask_zero=True)

    # train a Long Short Term Memory network followed by Fully Connected layers
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = LSTM(512, dropout_W=0.1, dropout_U=0.1, return_sequences=False)(embedded_sequences)
    x = Dense(512, activation='linear')(x)
    x = PReLU()(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Dense(512, activation='linear')(x)
    x = PReLU()(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    preds = Dense(3, activation='softmax')(x)

    model = Model(sequence_input, preds)
    return model

if __name__ == '__main__':
    train_keras_model_cv(gen_model, 'lstm/lstm-model', max_nb_words=MAX_NB_WORDS,
                         max_sequence_length=MAX_SEQUENCE_LENGTH, k_folds=5,
                         nb_epoch=50)
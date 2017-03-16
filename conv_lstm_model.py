import numpy as np
import os

from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Dropout, merge, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers import Embedding, LSTM, Conv1D, MaxPooling1D, GlobalMaxPooling1D
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

concat_axis = 1 if K.image_dim_ordering() == 'th' else -1

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
    embedding_layer_lstm = Embedding(nb_words,
                                     EMBEDDING_DIM,
                                     weights=[embedding_matrix],
                                     input_length=MAX_SEQUENCE_LENGTH,
                                     trainable=False, mask_zero=True)

    embedding_layer_conv = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    # train a Long Short Term Memory network followed by Fully Connected layers
    sequence_input_lstm = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer_lstm(sequence_input_lstm)
    x = LSTM(512, dropout_W=0.1, dropout_U=0.1, return_sequences=False, init='he_uniform')(embedded_sequences)
    x = PReLU(alpha=0.1)(x)

    sequence_input_conv = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_sequences_conv = embedding_layer_conv(sequence_input_conv)

    y1 = Conv1D(512, 5, border_mode='same')(embedding_sequences_conv)
    y1 = PReLU()(y1)

    y2 = Conv1D(512, 3, border_mode='same')(embedding_sequences_conv)
    y2 = PReLU()(y2)

    y3 = Conv1D(512, 4, border_mode='same')(embedding_sequences_conv)
    y3 = PReLU()(y3)

    y4 = Conv1D(512, 7, border_mode='same')(embedding_sequences_conv)
    y4 = PReLU()(y4)

    y = merge([y1, y2, y3, y4], mode='concat', concat_axis=concat_axis)
    y = GlobalMaxPooling1D()(y)

    m = merge([x, y], mode='concat', concat_axis=concat_axis)

    x = Dense(512, activation='linear')(m)
    x = PReLU()(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Dense(256, activation='linear')(x)
    x = PReLU()(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    preds = Dense(3, activation='softmax')(x)

    model = Model([sequence_input_lstm, sequence_input_conv], preds)
    return model

if __name__ == '__main__':
    train_keras_model_cv(gen_model, 'conv_lstm/conv_lstm-model', max_nb_words=MAX_NB_WORDS,
                         max_sequence_length=MAX_SEQUENCE_LENGTH, k_folds=5,
                         nb_epoch=50)


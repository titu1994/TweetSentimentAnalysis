import numpy as np
import glob
from sklearn_utils import evaluate

from keras.layers import Dense, Input, Dropout, merge
from keras.layers import Conv1D, Embedding, GlobalMaxPooling1D
from keras.models import Model

from keras import backend as K

from keras_utils import load_both, load_embedding_matrix, prepare_tokenized_data, train_keras_model_cv, prepare_data

MAX_NB_WORDS = 95000
MAX_SEQUENCE_LENGTH = 65
VALIDATION_SPLIT = 0.1
EMBEDDING_DIM = 300

EMBEDDING_DIR = 'embedding'
EMBEDDING_TYPE = 'glove.840B.300d.txt' # 'glove.6B.%dd.txt' % (EMBEDDING_DIM)

concat_axis = 1 if K.image_dim_ordering() == 'th' else -1

texts, labels, label_map = load_both()

data, word_index = prepare_tokenized_data(texts, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)

# prepare embedding matrix
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = load_embedding_matrix(EMBEDDING_DIR + "/" + EMBEDDING_TYPE,
                                          word_index, MAX_NB_WORDS, EMBEDDING_DIM)

def gen_n_conv_model():

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False, mask_zero=False)

    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    x = Conv1D(512, 5, activation='relu', border_mode='same', init='he_uniform')(embedded_sequences)
    #x = MaxPooling1D(3)(x)

    y = Conv1D(512, 3, activation='relu', border_mode='same', init='he_uniform')(embedded_sequences)
    #y = MaxPooling1D(3)(y)

    w = Conv1D(512, 4, activation='relu', border_mode='same', init='he_uniform')(embedded_sequences)
    #w = MaxPooling1D(3)(w)

    z = Conv1D(512, 7, activation='relu', border_mode='same', init='he_uniform')(embedded_sequences)
    #z = MaxPooling1D(3)(z)

    m = merge([x, y, z, w], mode='concat', concat_axis=concat_axis)

    x = GlobalMaxPooling1D()(m)
    #x = Flatten()(m)

    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)

    preds = Dense(3, activation='softmax')(x)

    model = Model(sequence_input, preds)
    return model


def write_predictions(model_dir='n_conv/', mode='train', dataset='full'):
    basepath = 'models/' + model_dir
    path = basepath + "*.h5"

    data, labels, texts, word_index = prepare_data(MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, mode=mode, dataset=dataset)
    files = glob.glob(path)

    nb_models = len(files)
    model_predictions = np.zeros((nb_models, data.shape[0], 3))

    model = gen_n_conv_model()

    for i, fn in enumerate(files):
        model.load_weights(fn)
        model_predictions[i, :, :] = model.predict(data, batch_size=100)

        print('Finished prediction for model %d' % (i + 1))

    if mode == 'train':
        np.save(basepath + "n_conv_predictions.npy", model_predictions)
    else:
        if dataset == 'full':
            save_dir = 'test'
        else:
            save_dir = dataset

        preds_save_path = save_dir + "/" + model_dir + "n_conv_predictions.npy"
        np.save(preds_save_path, model_predictions)


def calculate_score(model_dir='n_conv/', base_dir='test/', dataset='full'):
    basepath = base_dir + model_dir
    path = basepath + "*.npy"

    data, labels, texts, word_index = prepare_data(MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, mode='test', dataset=dataset)
    files = glob.glob(path)

    model_predictions = np.load(files[0])
    print('Loaded predictions. Shape = ', model_predictions.shape)

    model_predictions = model_predictions.mean(axis=0)

    preds = np.argmax(model_predictions, axis=1)
    evaluate(labels, preds)

if __name__ == '__main__':
    # train_keras_model_cv(gen_n_conv_model, 'n_conv/n_conv-model', max_nb_words=MAX_NB_WORDS,
    #                      max_sequence_length=MAX_SEQUENCE_LENGTH, k_folds=10,
    #                      nb_epoch=50)

    # write_predictions(mode='train')
    # write_predictions(mode='test')
    #write_predictions(mode='test', dataset='obama')
    #write_predictions(mode='test', dataset='romney')

    #calculate_score()
    calculate_score(base_dir='obama/', dataset='obama')
    calculate_score(base_dir='romney/', dataset='romney')


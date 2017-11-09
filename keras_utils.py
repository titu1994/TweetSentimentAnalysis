import numpy as np
import os
import pickle
import glob
import ast

np.random.seed(1000)

from sklearn_utils import load_both, load_obama, load_romney
from sklearn.model_selection import StratifiedKFold

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras import backend as K

train_obama_path = "data/obama_csv.csv"
train_romney_path = "data/romney_csv.csv"

train_obama_full_path = "data/full_obama_csv.csv"
train_romney_full_path = "data/full_romney_csv.csv"

test_obama_path = "data/obama_csv_test.csv"
test_romney_path = "data/romney_csv_test.csv"

# list of all models and their corresponding directories
model_dirs = ['conv/', 'n_conv/', 'lstm/', 'bidirectional_lstm/', 'multiplicative_lstm/']


def fbeta_score(y_true, y_pred):
    '''
    Computes the fbeta score. For ease of use, beta is set to 1.
    Therefore always computes f1_score
    '''
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))


def load_embedding_matrix(embedding_path, word_index, max_nb_words, embedding_dim, print_error_words=True):
    '''
    Either loads the created embedding matrix at run time, or uses the
    GLoVe 840B word embedding to create a mini initialized embedding matrix
    for use by Keras Embedding layers

    Args:
        embedding_path: path to the 840B word GLoVe Embeddings
        word_index: indices of all the words in the current corpus
        max_nb_words: maximum number of words in corpus
        embedding_dim: the size of the embedding dimension
        print_error_words: Optional, allows to print words from GLoVe
            that could not be parsed correctly.

    Returns:
        An Embedding matrix in numpy format
    '''
    if not os.path.exists('data/embedding_matrix max words %d embedding dim %d.npy' % (max_nb_words, embedding_dim)):
        embeddings_index = {}
        error_words = []

        print("Creating embedding matrix")
        print("Loading : ", embedding_path)

        # read the entire GLoVe embedding matrix
        f = open(embedding_path, encoding='utf8')
        for line in f:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            except Exception:
                error_words.append(word)

        f.close()

        # check for words that could not be loaded properly
        if len(error_words) > 0:
            print("%d words could not be added." % (len(error_words)))
            if print_error_words:
                print("Words are : \n", error_words)

        print('Preparing embedding matrix.')

        # prepare embedding matrix
        nb_words = min(max_nb_words, len(word_index))
        embedding_matrix = np.zeros((nb_words, embedding_dim))
        for word, i in word_index.items():
            if i >= nb_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        # save the constructed embedding matrix in a file for efficient loading next time
        np.save('data/embedding_matrix max words %d embedding dim %d.npy' % (max_nb_words,
                                                                             embedding_dim),
                embedding_matrix)

        print('Saved embedding matrix')

    else:
        # load pre-built embedding matrix
        embedding_matrix = np.load('data/embedding_matrix max words %d embedding dim %d.npy' % (max_nb_words,
                                                                                                embedding_dim))

        print('Loaded embedding matrix')

    return embedding_matrix


def create_ngram_set(input_list, ngram_value=2):
    # construct n-gram text from uni-gram text input
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


def prepare_tokenized_data(texts, max_nb_words, max_sequence_length, ngram_range=2):
    '''
    Tokenize the data from sentences to list of words

    Args:
        texts: sentences list
        max_nb_words: maximum vocabulary size in text corpus
        max_sequence_length: maximum length of sentence
        ngram_range: n-gram of sentences

    Returns:
        A list of tokenized sentences and the word index list which
        maps words to an integer index.
    '''
    if not os.path.exists('data/tokenizer.pkl'): # check if a prepared tokenizer is available
        tokenizer = Tokenizer(num_words=max_nb_words)  # if not, create a new Tokenizer
        tokenizer.fit_on_texts(texts)  # prepare the word index map

        with open('data/tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)  # save the prepared tokenizer for fast access next time

        print('Saved tokenizer.pkl')
    else:
        with open('data/tokenizer.pkl', 'rb') as f:  # simply load the prepared tokenizer
            tokenizer = pickle.load(f)
            print('Loaded tokenizer.pkl')

    sequences = tokenizer.texts_to_sequences(texts)  # transform text into integer indices lists
    word_index = tokenizer.word_index  # obtain the word index map
    print('Found %s unique 1-gram tokens.' % len(word_index))

    ngram_set = set()
    for input_list in sequences:
        for i in range(2, ngram_range + 1):  # prepare the n-gram sentences
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than max_features in order
    # to avoid collision with existing features.
    start_index = max_nb_words + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}
    word_index.update(token_indice)

    max_features = np.max(list(indice_token.keys())) + 1  # compute maximum number of n-gram "words"
    print('Now there are:', max_features, 'features')

    # Augmenting X_train and X_test with n-grams features
    sequences = add_ngram(sequences, token_indice, ngram_range)  # add n-gram features to original dataset
    print('Average sequence length: {}'.format(np.mean(list(map(len, sequences)), dtype=int)))  # compute average sequence length
    print('Max sequence length: {}'.format(np.max(list(map(len, sequences)))))  # compute maximum sequence length

    data = pad_sequences(sequences, maxlen=max_sequence_length)  # pad the sequence to the user defined max length

    return (data, word_index)


def train_keras_model_cv(model_gen, model_fn, max_nb_words=16000, max_sequence_length=140,
                         k_folds=3, nb_epoch=40, batch_size=100, seed=1000):
    '''
    Trains a provided Keras model with Stratified Cross Validation.

    Args:
        model_gen: a function which returns a Keras model
        model_fn: a string file name for the model to serialize the weights
        max_nb_words: maximum number of words in embedding
        max_sequence_length: maximum user defined sequence length
        k_folds: number of folds to train
        nb_epoch: number of epochs of training
        batch_size: batchsize of training each epoch
        seed: random seed for Stratified KFold. Keras ops are inherently
            non-deterministic due to use of CUDA and cuDNN to train models.
    '''
    data, labels, texts, word_index = prepare_data(max_nb_words, max_sequence_length)  # load the text dataset

    print("Dataset :", data.shape)
    skf = StratifiedKFold(k_folds, shuffle=True, random_state=seed)  # initialize Stratified Fold Generator

    fbeta_scores = []

    for i, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):  # for each fold
        x_train, y_train = data[train_idx, :], labels[train_idx]  # obtain the Train samples and labels
        x_test, y_test = data[test_idx, :], labels[test_idx]  # obtain the Test samples and labels

        y_train_categorical = to_categorical(np.asarray(y_train))  # convert to one-hot representation
        y_test_categorical = to_categorical(np.asarray(y_test))  # convert to one-hot representation

        K.clear_session()  # reset GPU memory

        # generate a new Keras model
        model = model_gen()  # type: Model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', fbeta_score])  # compile model

        # save model which obtains best validation f1-score
        model_checkpoint = ModelCheckpoint('models/%s-cv-%d.h5' % (model_fn, i + 1), monitor='val_fbeta_score',
                                           verbose=2,
                                           save_weights_only=True,
                                           save_best_only=True, mode='max')

        # reduce learning rate as schedule demands
        reduce_lr = ReduceLROnPlateau(monitor='val_fbeta_score', patience=5, mode='max',
                                      factor=0.8, cooldown=5, min_lr=1e-6, verbose=2)

        # train model
        model.fit(x_train, y_train_categorical, validation_data=(x_test, y_test_categorical),
                  callbacks=[model_checkpoint, reduce_lr], epochs=nb_epoch, batch_size=batch_size)

        # load the saved best weights
        model.load_weights('models/%s-cv-%d.h5' % (model_fn, i + 1))

        # evaluate final performance of the model on test set
        scores = model.evaluate(x_test, y_test_categorical, batch_size=batch_size)

        # save the f1 score for final averaging
        fbeta_scores.append(scores[-1])

        print('\nF1 Scores of Cross Validation %d: %0.4f' % (i + 1, scores[-1]))

        # delete Keras model from CPU memory
        del model

    # compute average f1 score over all folds
    print("Average fbeta score : ", sum(fbeta_scores) / len(fbeta_scores))

    # save the f1 score results per fold into a file
    with open('models/%s-scores.txt' % (model_fn), 'w') as f:
        f.write(str(fbeta_scores))


def prepare_data(max_nb_words, max_sequence_length, mode='train', dataset='full'):
    '''
    Loads the appropriate dataset as required

    Args:
        max_nb_words: maximum vocabulary size
        max_sequence_length: maximum length of a sentence
        mode: decided which dataset to load. Can be one of
            'train' or 'test'.
        dataset: decides which dataset to load.
            Can be one of :
            -   'full' (for Joint Training)
            -   'obama' (for just Obama dataset)
            -   'romney' (for just Romney dataset)


    Returns:
        The preprocessed text data, labels, the raw text sentences and the word indices
    '''
    assert dataset in ['full', 'obama', 'romney']

    print('Loading %s data' % mode)

    if dataset == 'full':
        texts, labels, label_map = load_both(mode)
    elif dataset == 'obama':
        texts, labels, label_map = load_obama(mode)
    else:
        texts, labels, label_map = load_romney(mode)

    print('Tokenizing texts')
    data, word_index = prepare_tokenized_data(texts, max_nb_words, max_sequence_length)
    print('Finished tokenizing texts')
    print('-' * 80)
    return data, labels, texts, word_index


def get_keras_scores(normalize_scores=False):
    '''
    Utility function for computing the scores of all the
    the Keras models from the serialized score lists.

    Args:
        normalize_scores: whether to normalize the scores
            Normalization is done by weighing of the sum of weights

    Returns:
        a list of classifier scores for all Keras models
    '''
    clf_scores = []

    for m, model_dir in enumerate(model_dirs):
        weights_path = 'models/' + model_dir + '*.txt'

        weight_path = glob.glob(weights_path)
        print('Loading score file [0]:', weight_path)

        with open(weight_path[0], 'r') as f:
            clf_weight_data = ast.literal_eval(f.readline())

        clf_scores.extend(clf_weight_data)

    if normalize_scores:
        weight_sum = np.sum(np.asarray(clf_scores, dtype=np.float32))
        weights = [w / weight_sum for w in clf_scores]
        clf_scores = weights

    return clf_scores


if __name__ == '__main__':
    max_nb_words = 90046
    max_sequence_length = 65

    data, labels, texts, word_index = prepare_data(max_nb_words, max_sequence_length)

    print(data.shape)
    print(data.dtype)
    print(data[0])
    print('\n', '*' * 80, '\n')
    print(data[1])
    pass

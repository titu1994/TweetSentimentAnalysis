import pandas as pd
import numpy as np
import joblib
import time
import os
import pickle

np.random.seed(1000)

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

train_obama_path = "data/obama_csv.csv"
train_romney_path = "data/romney_csv.csv"

train_obama_full_path = "data/full_obama_csv.csv"
train_romney_full_path = "data/full_romney_csv.csv"

if not os.path.exists('models/'):
    os.makedirs('models/')

subdirs = ['conv/', 'conv_lstm/', 'lstm/', 'gru/', 'n_conv/', 'xgboost/', 'mnb/', 'svm/', 'nbsvm/']

for sub in subdirs:
    path = 'models/' + sub

    if not os.path.exists(path):
        os.makedirs(path)

def load_both(use_full_data=False):
    if use_full_data:
        obama_df = pd.read_csv(train_obama_full_path, sep='\t', encoding='latin1')
    else:
        obama_df = pd.read_csv(train_obama_path, sep='\t', encoding='latin1')
    # Remove rows who have no class label attached, can hand label later
    obama_df = obama_df[pd.notnull(obama_df['label'])]
    obama_df['label'] = obama_df['label'].astype(np.int)

    if use_full_data:
        romney_df = pd.read_csv(train_romney_full_path, sep='\t', encoding='latin1')
    else:
        romney_df = pd.read_csv(train_romney_path, sep='\t', encoding='latin1')
    # Remove rows who have no class label attached, can hand label later
    romney_df = romney_df[pd.notnull(romney_df['label'])]
    romney_df['label'] = romney_df['label'].astype(np.int)

    texts = []  # list of text samples
    labels_index = {-1: 0,
                    0: 1,
                    1: 2}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids

    obama_df = obama_df[obama_df['label'] != 2]  # drop all rows with class = 2
    romney_df = romney_df[romney_df['label'] != 2]  # drop all rows with class = 2

    nb_rows = len(obama_df)
    for i in range(nb_rows):
        row = obama_df.iloc[i]
        texts.append(str(row['tweet']))
        labels.append(labels_index[int(row['label'])])

    nb_rows = len(romney_df)
    for i in range(nb_rows):
        row = romney_df.iloc[i]
        texts.append(str(row['tweet']))
        labels.append(labels_index[int(row['label'])])

    texts = np.asarray(texts)
    labels = np.asarray(labels)

    return texts, labels, labels_index


def tokenize(texts):
    if os.path.exists('data/vectorizer.pkl'):
        with open('data/vectorizer.pkl', 'rb') as f:
            tfidf_vect = pickle.load(f)
            x_counts = tfidf_vect.transform(texts)
    else:
        tfidf_vect = CountVectorizer(ngram_range=(1, 2))
        x_counts = tfidf_vect.fit_transform(texts)

        with open('data/vectorizer.pkl', 'wb') as f:
            pickle.dump(tfidf_vect, f)

    print('Shape of tokenizer counts : ', x_counts.shape)
    return x_counts


def tfidf(x_counts):
    if os.path.exists('data/tfidf.pkl'):
        with open('data/tfidf.pkl', 'rb') as f:
            transformer = pickle.load(f)
            x_tfidf = transformer.transform(x_counts)
    else:
        transformer = TfidfTransformer()
        x_tfidf = transformer.fit_transform(x_counts)

        with open('data/tfidf.pkl', 'wb') as f:
            pickle.dump(transformer, f)

    return x_tfidf


def train_sklearn_model_cv(model_gen, model_fn, use_full_data=False, k_folds=3, seed=1000):
    data, labels = prepare_data(use_full_data)

    skf = StratifiedKFold(k_folds, shuffle=True, random_state=seed)

    fbeta_scores = []

    for i, (train_idx, test_idx) in enumerate(skf.split(data, labels)):
        x_train, y_train = data[train_idx, :], labels[train_idx]
        x_test, y_test = data[test_idx, :], labels[test_idx]

        print('\nBegin training classifier %d' % (i + 1), 'Training samples : ', x_train.shape[0],
              'Testing samples : ', x_test.shape[0])

        model = model_gen()

        t1 = time.time()
        try:
            model.fit(x_train, y_train)
        except TypeError:
            # Model does not support sparce matrix input, convert to dense matrix input
            x_train = x_train.toarray()
            model.fit(x_train, y_train)

            # dense matrix input is very large, delete to preserve memory
            del(x_train)

        t2 = time.time()

        print('Classifier %d training time : %0.3f seconds.' % (i + 1, t2 - t1))

        print('Begin testing classifier %d' % (i + 1))

        t1 = time.time()
        try:
            preds = model.predict(x_test)
        except TypeError:
            # Model does not support sparce matrix input, convert to dense matrix input
            x_test = x_test.toarray()
            preds = model.predict(x_test)

            # dense matrix input is very large, delete to preserve memory
            del(x_test)

        t2 = time.time()

        print('Classifier %d finished predicting in %0.3f seconds.' % (i + 1, t2 - t1))

        f1score  = f1_score(y_test, preds, average='micro')
        fbeta_scores.append(f1score)

        print('\nF1 Scores of Estimator %d: %0.4f' % (i + 1, f1score))

        joblib.dump(model, 'models/%s-cv-%d.pkl' % (model_fn, i + 1))

        del model

    print("\nAverage fbeta score : ", sum(fbeta_scores) / len(fbeta_scores))

    with open('models/%s-scores.txt' % (model_fn), 'w') as f:
        f.write(str(fbeta_scores))


def prepare_data(use_full_data=False):
    print('Loading data')
    texts, labels, label_map = load_both(use_full_data)
    print('Tokenizing texts')
    x_counts = tokenize(texts)
    print('Finished tokenizing texts')
    data = tfidf(x_counts)
    print('Finished computing TF-IDF')
    print('-' * 80)
    return data, labels


if __name__ == '__main__':
    pass
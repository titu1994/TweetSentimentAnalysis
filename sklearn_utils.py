import pandas as pd
import numpy as np
import joblib
import time
import os
import pickle

from keras_utils import load_both

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

if not os.path.exists('models/xgboost/'):
    os.makedirs('models/xgboost/')

if not os.path.exists('models/mnb/'):
    os.makedirs('models/mnb/')

if not os.path.exists('models/svm/'):
    os.makedirs('models/svm/')


def tokenize(texts, stop_words=None):
    if os.path.exists('data/vectorizer.pkl'):
        tfidf_vect = pickle.load('data/vectorizer.pkl')
        x_counts = tfidf_vect.transform(texts)
    else:
        tfidf_vect = TfidfVectorizer(stop_words=stop_words, sublinear_tf=True)
        x_counts = tfidf_vect.fit_transform(texts)

        with open('data/vectorizer.pkl', 'wb') as f:
            pickle.dump(tfidf_vect, f)

    print('Shape of tokenizer counts : ', x_counts.shape)
    return x_counts


def tfidf(x_counts):
    if os.path.exists('data/tfidf.pkl'):
        transformer = pickle.load('data/tfidf.pkl')
        x_tfidf = transformer.transform(x_counts)
    else:
        transformer = TfidfTransformer()
        x_tfidf = transformer.fit_transform(x_counts)

        with open('data/tfidf.pkl', 'wb') as f:
            pickle.dump(transformer, f)

    return x_tfidf


def train_keras_model_cv(model, model_fn, k_folds=3, seed=1000):
    texts, labels, label_map = load_both()
    x_counts = tokenize(texts)
    data = tfidf(x_counts)

    skf = StratifiedKFold(k_folds, shuffle=True, random_state=seed)

    fbeta_scores = []

    for i, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):
        x_train, y_train = data[train_idx, :], labels[train_idx]
        x_test, y_test = data[test_idx, :], labels[test_idx]

        print('Begin training classifier %d' % (i + 1))

        t1 = time.time()
        model.fit(x_train, y_train)
        t2 = time.time()

        print('Classifier %d training time : %0.3f seconds.' % (i + 1, t2 - t1))

        print('Begin testing classifier %d' % (i + 1))

        t1 = time.time()
        preds = model.predict(x_test, y_test)
        t2 = time.time()

        print('Classifier %d finished predicting in %0.3f seconds.' % (i + 1, t2 - t1))

        f1score  = f1_score(y_test, preds)
        fbeta_scores.append(f1score)

        print('\nF1 Scores of Estimator %d: %0.4f' % (i + 1, f1score))

        joblib.dump(model, 'models/%s-cv-%d.h5' % (model_fn, i + 1))

        del model

    print("Average fbeta score : ", sum(fbeta_scores) / len(fbeta_scores))

    with open('models/%s-scores.txt' % (model_fn), 'w') as f:
        f.write(str(fbeta_scores))


if __name__ == '__main__':
    pass
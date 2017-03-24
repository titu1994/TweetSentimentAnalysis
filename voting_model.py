from sklearn_utils import *
import numpy as np
import glob
import joblib
import ast
import re

from voting_ensemble import SoftVoteClassifier
from sklearn.metrics import f1_score


clfs, clf_weights = load_trained_models(normalize_weights=True)

def scoring(estimator, X, y):
    preds = estimator.predict(X)
    return f1_score(y, preds, average='micro')

def fit_voting_classifier():
    model = SoftVoteClassifier(clfs, weights=None)

    np.random.seed(1000)
    print('Loading data')
    texts, labels, label_map = load_both()
    print('Tokenizing texts')
    x_counts = tokenize(texts)
    print('Finished tokenizing texts')
    data = tfidf(x_counts)
    print('Finished computing TF-IDF')

    preds = model.predict(data)

    score = f1_score(labels, preds, average='micro')
    print('\nF1 Score : ', score)

    print('Saving predictions')
    np.save('vote_model/voting_predictions.npy', preds)


if __name__ == '__main__':
    fit_voting_classifier()
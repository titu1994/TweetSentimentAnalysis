from sklearn_utils import load_trained_sklearn_models, prepare_data, evaluate
from keras_utils import get_keras_scores
import numpy as np
import glob
import joblib
import ast
import re

from voting_ensemble import SoftVoteClassifier
from sklearn.metrics import f1_score

clfs, sklearn_scores = load_trained_sklearn_models()
keras_scores = get_keras_scores(normalize_scores=False)

clf_weights = []

for i in range(len(sklearn_scores) // 100):
    score_sum = 0.0

    for j in range(100):
        loc = i * 100 + j
        score = sklearn_scores[loc]
        score_sum += score

    clf_weights.append(score_sum / 100.)

for i in range(len(keras_scores) // 10):
    score_sum = 0.0

    for j in range(10):
        loc = i * 10 + j
        score = keras_scores[loc]
        score_sum += score

    clf_weights.append(score_sum / 10.)

score_sum_ = sum(clf_weights)
clf_weights = [s / score_sum_ for s in clf_weights]

model = SoftVoteClassifier(None, weights=clf_weights)

def fit_voting_classifier(dataset='full'):
    np.random.seed(1000)
    # print('Loading data')
    data, labels = prepare_data(mode='test', dataset=dataset)

    if dataset == 'full':
        pred_dir = 'test/*/'
    elif dataset == 'obama':
        pred_dir = 'obama/*/'
    else:
        pred_dir = 'romney/*/'

    preds = model.predict_proba_dir(pred_dir)

    evaluate(labels, np.argmax(preds, axis=1))

    # print('Saving predictions')
    # np.save('test/voting/voting_predictions.npy', preds)
    #
    # data, labels = prepare_data()
    #
    # model = SoftVoteClassifier(clfs, weights=None)
    # preds = model.predict_proba(data)
    # print(preds.shape)
    # np.save('models/voting/voting_predictions.npy', preds)
    #
    # print('VotingClassifier fit complete!')

if __name__ == '__main__':
    #fit_voting_classifier()

    fit_voting_classifier(dataset='obama')
    fit_voting_classifier(dataset='romney')
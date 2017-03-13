from sklearn_utils import *

import numpy as np
np.random.seed(1000)
import joblib
import time

import xgboost as xgb
from xgboost import Booster

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

def generate_cv_dmatrix(data, labels, k_folds=10, seed=1000):
    skf = StratifiedKFold(k_folds, shuffle=True, random_state=seed)

    for i, (train_idx, test_idx) in enumerate(skf.split(data, labels)):
        x_train, y_train = data[train_idx, :], labels[train_idx]
        x_test, y_test = data[test_idx, :], labels[test_idx]

        train_dmatrix = xgb.DMatrix(x_train, y_train)
        test_dmatrix = xgb.DMatrix(x_test, y_test)

        yield train_dmatrix, test_dmatrix, y_test


def train_xgboost(iters=100, k_folds=10, use_full_data=False, seed=1000):
    print('Loading data')
    texts, labels, label_map = load_both(use_full_data)
    print('Tokenizing texts')
    x_counts = tokenize(texts)
    print('Finished tokenizing texts')
    data = tfidf(x_counts)
    print('Finished computing TF-IDF')
    print('-' * 80)

    params = {
        'booster': 'gblinear',
        'objective': 'multi:softprob',
        'eta': 0.01,
        'max_depth': 10,
        'subsample': 0.95,
        'lambda': 1e-3,
        'updater': 'grow_gpu',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'seed': 1000,
       }

    f1_scores = []

    for i, (train, val, val_labels) in enumerate(generate_cv_dmatrix(data, labels, k_folds, seed)):
        print('\nBegin training classifier %d' % (i + 1))
        t1 = time.time()

        model = xgb.train(params, train, num_boost_round=iters,
                          evals=[(val, 'val')], verbose_eval=False) # type: Booster

        t2 = time.time()
        print('Classifier %d training time : %0.3f seconds.' % (i + 1, t2 - t1))

        print('Begin testing classifier %d' % (i + 1))
        t1 = time.time()

        preds = model.predict(val)
        preds = np.argmax(preds, axis=1)

        t2 = time.time()
        print('Classifier %d finished predicting in %0.3f seconds.' % (i + 1, t2 - t1))

        f1score = f1_score(val_labels, preds, average='micro')
        f1_scores.append(f1score)

        print('\nF1 Scores of Estimator %d: %0.4f' % (i + 1, f1score))

        joblib.dump(model, 'models/xgboost/xgb-model-cv-%d.pkl' % (i + 1))
        del model

    print("\nAverage fbeta score : ", sum(f1_scores) / len(f1_scores))

    with open('models/%s-scores.txt' % ('xgboost/xgb-model'), 'w') as f:
        f.write(str(f1_scores))


def scoring(estimator, X, y):
    preds = estimator.predict(X)
    return f1_score(y, preds, average='micro')


def param_search():
    params = {'alpha' : np.linspace(0.6, 0.8, num=100)}
    print('Params : ', params)

    model = xgb.XGBClassifier(max_depth=4)
    g = GridSearchCV(model, param_grid=params, scoring=scoring,
                     n_jobs=-1, cv=100, verbose=1)

    np.random.seed(1000)
    print('Loading data')
    texts, labels, label_map = load_both()
    print('Tokenizing texts')
    x_counts = tokenize(texts)
    print('Finished tokenizing texts')
    data = tfidf(x_counts)
    print('Finished computing TF-IDF')

    g.fit(data, labels)
    print("Best parameters set found on development set:")
    print()
    print(g.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = g.cv_results_['mean_test_score']
    stds = g.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, g.cv_results_['params']):
        print("%0.6f (+/-%0.06f) for %r"
              % (mean, std * 2, params))


if __name__ == '__main__':
    train_xgboost(iters=100, k_folds=100)

    #param_search()
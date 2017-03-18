from sklearn_utils import *

import numpy as np
np.random.seed(1000)
import joblib
import time
import glob

import xgboost as xgb
from xgboost import Booster

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score


params = {
    'booster': 'gblinear',
    'objective': 'multi:softprob',
    'eta': 0.01,
    'max_depth': 10,
    'subsample': 1.0,
    'lambda': 1e-2,
    'updater': 'grow_gpu',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'seed': 1000,
}


def generate_cv_dmatrix(data, labels, k_folds=10, seed=1000):
    skf = StratifiedKFold(k_folds, shuffle=True, random_state=seed)

    for i, (train_idx, test_idx) in enumerate(skf.split(data, labels)):
        x_train, y_train = data[train_idx, :], labels[train_idx]
        x_test, y_test = data[test_idx, :], labels[test_idx]

        train_dmatrix = xgb.DMatrix(x_train, y_train)
        test_dmatrix = xgb.DMatrix(x_test, y_test)

        yield train_dmatrix, test_dmatrix, y_test


def train_xgboost(iters=100, k_folds=10, use_full_data=False, seed=1000):
    data, labels = prepare_data(use_full_data)

    f1_scores = []

    for i, (train, val, val_labels) in enumerate(generate_cv_dmatrix(data, labels, k_folds, seed)):
        print('\nBegin training classifier %d' % (i + 1))
        t1 = time.time()

        model = xgb.train(params, train, num_boost_round=iters,
                          evals=[(val, 'val')], verbose_eval=False,
                          early_stopping_rounds=50) # type: Booster

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


def eval_func(y_true, y_pred):
    pred = y_pred.get_label()
    y_true = np.argmax(y_true, axis=1)
    return 'f1_score', f1_score(y_true, pred, average='micro')


def train_full_model(iters=100, use_full_data=False, seed=1000):
    np.random.seed(seed)
    data, labels = prepare_data(use_full_data)
    train_dmatrix = xgb.DMatrix(data, labels)

    print('\nBegin training classifier')
    t1 = time.time()

    model = xgb.train(params, train_dmatrix, num_boost_round=iters,
                      feval=eval_func, maximize=True, early_stopping_rounds=50,
                      evals=[(train_dmatrix, 'train'),], verbose_eval=True)  # type: Booster

    t2 = time.time()
    print('Classifier training time : %0.3f seconds.' % (t2 - t1))

    print('Begin testing classifier ')
    t1 = time.time()

    preds = model.predict(train_dmatrix)
    preds = np.argmax(preds, axis=1)

    t2 = time.time()
    print('Classifier finished predicting in %0.3f seconds.' % (t2 - t1))

    f1score = f1_score(labels, preds, average='micro')

    print('\nF1 Scores of Estimator: %0.4f' % (f1score))
    joblib.dump(model, 'models/xgboost/xgb-model-final.pkl')

    print('Finished saving model')

def scoring(estimator, X, y):
    preds = estimator.predict(X)
    return f1_score(y, preds, average='micro')


def write_predictions(model_dir='xgboost/'):
    basepath = 'models/' + model_dir
    path = basepath + "*.pkl"

    data, labels = prepare_data()
    nb_features = data.shape[0]

    data = xgb.DMatrix(data, labels)
    files = glob.glob(path)

    nb_models = len(files)

    model_predictions = np.zeros((nb_models, nb_features, 3))

    for i, fn in enumerate(files):
        model = joblib.load(fn) # type: Booster
        model_predictions[i, :, :] = model.predict(data)
        print('Finished prediction for model %d' % (i + 1))

    np.save(basepath + "xgboost_predictions.npy", model_predictions)

if __name__ == '__main__':
    #train_xgboost(iters=1000, k_folds=100)
    #param_search()
    write_predictions()
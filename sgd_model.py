from sklearn_utils import *
import numpy as np
import glob
import joblib
import multiprocessing

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score


def proba_model_gen():
    n_iter = np.ceil((10 ** 6) / 10000) # 100 iterations
    model = SGDClassifier(alpha=0.0001, n_iter=n_iter, loss="log")
    return model

def model_gen():
    n_iter = np.ceil((10 ** 6) / 10000) # 100 iterations
    model = SGDClassifier(alpha=0.0001, n_iter=n_iter)
    return model

def scoring(estimator, X, y):
    preds = estimator.predict(X)
    return f1_score(y, preds, average='micro')

def param_search():
    n_iter = np.ceil((10 ** 6) / 10000)
    params = {'alpha' : 10.0 ** -np.arange(1,7),
              'loss' : ['log', 'modified_huber'],
              }
    print('Params : ', params)

    model = SGDClassifier(n_iter=n_iter)

    # use 1 less core than available, prevents locking up of laptop
    n_cores = multiprocessing.cpu_count() - 1

    g = GridSearchCV(model, param_grid=params, scoring=scoring,
                     n_jobs=n_cores, cv=100, verbose=1)

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


def write_predictions(model_dir='sgd/'):
    basepath = 'models/' + model_dir + 'sgd-model-'
    path = basepath + "*.pkl"

    data, labels = prepare_data()
    files = glob.glob(path)

    nb_models = len(files)

    model_predictions = np.zeros((nb_models, data.shape[0], 3))

    for i, fn in enumerate(files):
        model = joblib.load(fn) # type: SGDClassifier

        model_predictions[i, :, :] = model.predict_proba(data)
        print('Finished prediction for model %d' % (i + 1))

    np.save(basepath + "sgd_predictions.npy", model_predictions)

if __name__ == '__main__':
    train_sklearn_model_cv(proba_model_gen, 'sgd/sgd-model', k_folds=100, use_full_data=False)
    train_full_model(proba_model_gen, 'sgd/sgd-model', use_full_data=False)
    train_sklearn_model_cv(model_gen, 'sgd/sgd-predict-model', k_folds=100, use_full_data=False)
    train_full_model(proba_model_gen, 'sgd/sgd-predict-model', use_full_data=False)
    #param_search()
    write_predictions()
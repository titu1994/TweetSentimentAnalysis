from sklearn_utils import *
import numpy as np
import glob
import joblib
import multiprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score


def model_gen():
    model = LogisticRegression(C=1.09)
    return model

def scoring(estimator, X, y):
    preds = estimator.predict(X)
    return f1_score(y, preds, average='micro')

def param_search():
    params = {'C' : np.linspace(1.0, 1.1, num=21),
             # 'beta' : [0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
              }
    print('Params : ', params)

    model = LogisticRegression()

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


def write_predictions(model_dir='logistic/'):
    basepath = 'models/' + model_dir
    path = basepath + "*.pkl"

    data, labels = prepare_data()
    files = glob.glob(path)

    nb_models = len(files)

    model_predictions = np.zeros((nb_models, data.shape[0], 3))

    for i, fn in enumerate(files):
        model = joblib.load(fn) # type: LogisticRegression

        model_predictions[i, :, :] = model.predict_proba(data)
        print('Finished prediction for model %d' % (i + 1))

    np.save(basepath + "logistic_predictions.npy", model_predictions)

if __name__ == '__main__':
    #train_sklearn_model_cv(model_gen, 'logistic/logistic-model', k_folds=100, use_full_data=False)
    #train_full_model(model_gen, 'logistic/logistic-model', use_full_data=False)
    #param_search()
    write_predictions()
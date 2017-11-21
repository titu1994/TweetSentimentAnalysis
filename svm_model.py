from sklearn_utils import *
import numpy as np
import glob
import joblib
import multiprocessing

from sklearn.svm import LinearSVC

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score


def model_gen():
    model = LinearSVC(C=0.152)
    return model

def scoring(estimator, X, y):
    preds = estimator.predict(X)
    return f1_score(y, preds, average='micro')

def param_search():
    params = {'C' : np.linspace(0.10, 0.2, num=51),
              }
    print('Params : ', params)

    model = LinearSVC()

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


def write_predictions(model_dir='svm/'):
    basepath = 'models/' + model_dir
    path = basepath + "*.pkl"

    data, labels = prepare_data()
    files = glob.glob(path)

    nb_models = len(files)

    model_predictions = np.zeros((nb_models, data.shape[0], 3))

    for i, fn in enumerate(files):
        model = joblib.load(fn) # type: LinearSVC

        model_predictions[i, :, :] = model._predict_proba_lr(data)
        print('Finished prediction for model %d' % (i + 1))

    np.save(basepath + "svm_predictions.npy", model_predictions)

if __name__ == '__main__':
    train_sklearn_model_cv(model_gen, 'svm/svm-model', k_folds=100)
    #param_search()
    write_predictions()

    evaluate_sklearn_model('svm/')
    evaluate_sklearn_model('svm/', dataset='obama')
    evaluate_sklearn_model('svm/', dataset='romney')
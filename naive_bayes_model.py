from sklearn_utils import *
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score


def model_gen():
    model = MultinomialNB(alpha=0.7)
    return model

def scoring(estimator, X, y):
    preds = estimator.predict(X)
    return f1_score(y, preds, average='micro')

def param_search():
    params = {'alpha' : np.linspace(0.6, 0.8, num=100)}
    print('Params : ', params)

    model = MultinomialNB()
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
    train_sklearn_model_cv(model_gen, 'mnb/mnb-model', k_folds=100, use_full_data=False)

    #param_search()
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
import operator
import glob
import warnings

def get_predictions(model, X):
    if hasattr(model, 'predict_proba'): # Normal SKLearn classifiers
        pred = model.predict_proba(X)
    elif hasattr(model, '_predict_proba_lr'): # SVMs
        pred = model._predict_proba_lr(X)
    else:
        pred = model.predict(X)

    if len(pred.shape) == 1:  # for 1-d ouputs
            pred = pred[:, None]

    return pred


def check_module_exists(modulename):
    try:
        __import__(modulename)
    except ImportError:
        return False
    return True

if check_module_exists('xgboost'):
    import xgboost as xgb


class SoftVoteClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier for pre-trained scikit-learn estimators.

    Parameters
    ----------

    clf : tuple - (classifier_name, clf)
      A list of pre-trained scikit-learn classifier objects.
      Can include XGBoost models at the very end of the classifier list.

    weights : `list` (default: `None`)
      If `None`, the majority rule voting will be applied to the predicted class labels.
        If a list of weights (`float` or `int`) is provided, the averaged raw probabilities (via `predict_proba`)
        will be used to determine the most confident class label.

    normalize_weights : bool (default: False)
      If True, will normalize the weights provided so that they sum up to 1.0

    verbose : bool (default: False)
      If True, will print out information about model prediction

    """
    def __init__(self, clfs, weights=None, normalize_weights=False, verbose=False):
        self.clfs = clfs
        self.verbose = verbose
        self.normalize_weights = normalize_weights

        if self.normalize_weights:
            weight_sum = np.sum(np.asarray(weights))
            weights = [w / weight_sum for w in weights]
            self.weights = weights
        else:
            self.weights = weights

    def fit(self, X, y):
        """
        Fit the scikit-learn estimators.

        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]
            Training data
        y : list or numpy array, shape = [n_samples]
            Class labels

        """
        warnings.warn('SoftVoteClassifier simply votes on pre-trained models. Please provide pre-trained model '
                      'in the classifier list.')

    def predict(self, X):
        """
        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]

        Returns
        ----------

        maj : list or numpy array, shape = [n_samples]
            Predicted class labels by majority rule

        """
        self.probas_ = []
        data = X
        for name, clf in self.clfs:
            if check_module_exists('xgboost'):
                if isinstance(clf, xgb.Booster): # XGBoost model, needs DMatrix for data
                    if not hasattr(data, 'feature_names'): # data is numpy array, convert to DMatrix
                        data = xgb.DMatrix(X)

            self.probas_.append(get_predictions(clf, data))

            if self.verbose:
                print('Obtained predictions of model %s' % name)

        avg = np.average(self.probas_, axis=0, weights=self.weights)

        # self.class_probas_ = np.asarray([get_predictions(clf, X) for clf in self.clfs])
        # if self.weights:
        #     #maj = np.apply_along_axis(lambda x: max(enumerate(x), key=operator.itemgetter(1))[0], axis=1, arr=self.class_probas_)
        #     avg = np.average(self.probas_, axis=0, weights=self.weights)
        # else:
        #     #maj = np.asarray([np.argmax(np.bincount(self.class_probas_[:, c])) for c in range(self.class_probas_.shape[1])])
        #

        maj = np.argmax(avg, axis=1)

        return maj

    def predict_proba(self, X):

        """
        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]

        Returns
        ----------

        avg : list or numpy array, shape = [n_samples, n_probabilities]
            Weighted average probability for each class per sample.

        """
        self.probas_ = []
        data = X
        for name, clf in self.clfs:
            if check_module_exists('xgboost'):
                if isinstance(clf, xgb.Booster): # XGBoost model, needs DMatrix for data
                    if not hasattr(data, 'feature_names'): # data is numpy array, convert to DMatrix
                        data = xgb.DMatrix(X)

            self.probas_.append(get_predictions(clf, data))

            if self.verbose:
                print('Obtained predictions of model %s' % name)

        avg = np.average(self.probas_, axis=0, weights=self.weights)

        return avg


    def predict_dir(self, dir):
        """
        Parameters
        ----------

        dir : a directory containing the numpy arrays which contain the predictions

        Returns
        ----------

        maj : list or numpy array, shape = [n_samples]
            Predicted class labels by majority rule

        """
        self.probas_ = []
        path = dir + "*.npy"
        files = glob.glob(path)

        for i, fn in enumerate(files):
            if 'voting' in fn:
                continue

            preds = np.load(fn)
            preds = preds.mean(axis=0)
            self.probas_.append(preds)

            if self.verbose:
                print('Obtained predictions of model %d' % (i + 1))

        avg = np.average(self.probas_, axis=0, weights=self.weights)
        maj = np.argmax(avg, axis=1)
        return maj

    def predict_proba_dir(self, dir):

        """
        Parameters
        ----------

        dir : a directory containing the numpy arrays which contain the predictions

        Returns
        ----------

        avg : list or numpy array, shape = [n_samples, n_probabilities]
            Weighted average probability for each class per sample.

        """
        self.probas_ = []
        path = dir + "*.npy"
        files = glob.glob(path)

        for i, fn in enumerate(files):
            if 'voting' in fn:
                continue

            preds = np.load(fn)
            preds = preds.mean(axis=0)
            self.probas_.append(preds)

            if self.verbose:
                print('Obtained predictions of model %d' % (i + 1))

        avg = np.average(self.probas_, axis=0, weights=self.weights)

        return avg
import numpy as np
import joblib
import glob
from copy import copy

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin

import keras.models as keras_models
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

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


class StackedGeneralizer(BaseEstimator, ClassifierMixin):
    """Base class for stacked generalization classifier models
    """

    def __init__(self, blending_models=None, n_folds=10, verbose=True):
        """
        Stacked Generalizer Classifier

        Trains a series of base models using K-fold cross-validation, then combines
        the predictions of each model into a set of features that are used to train
        a high-level classifier model.
        Parameters
        -----------
        blending_model: object
            A classifier model used to aggregate the outputs of the trained base
            models. Must have a .fit and .predict_proba/.predict method
        n_folds: int
            The number of K-folds to use in =cross-validated model training
        verbose: boolean
        """
        self.blending_models = blending_models
        self.n_folds = n_folds
        self.verbose = verbose

    def fit(self, X_indices, y):
        X_blend = self._fitTransformBaseModels()

        if X_indices is not None:
            self._fitBlendingModel(X_blend[X_indices], y)
        else:
            self._fitBlendingModel(X_blend, y)

    def predict(self, pred_directory, X_indices=None):
        # perform model averaging to get predictions
        predictions_dir = pred_directory if pred_directory is not None else 'models/*/'

        X_blend = self.transformBaseModels(predictions_dir)

        if X_indices is not None:
            predictions = self._transformBlendingModel(X_blend[X_indices])
        else:
            predictions = self._transformBlendingModel(X_blend)

        pred_classes = np.argmax(predictions, axis=1)
        return pred_classes

    def predict_proba(self, pred_directory, X_indices=None):
        # perform model averaging to get predictions
        predictions_dir = pred_directory if pred_directory is not None else 'models/*/'

        X_blend = self.transformBaseModels(predictions_dir)

        if X_indices is not None:
            predictions = self._transformBlendingModel(X_blend[X_indices])
        else:
            predictions = self._transformBlendingModel(X_blend)

        return predictions

    def evaluate(self, y, y_pred):
        print(classification_report(y, y_pred))
        print('Confusion Matrix:')
        print(confusion_matrix(y, y_pred))
        return(accuracy_score(y, y_pred))

    def transformBaseModels(self, pred_dir='models/*/'):
        # predict via model averaging
        predictions = []

        base_path = pred_dir
        path = base_path + "*.npy"

        files = glob.glob(path)
        for file in files:
            if self.verbose: print('Loading numpy file %s' % (file))
            cv_predictions = np.load(file)
            predictions.append(cv_predictions.mean(axis=0)) # take mean on all cv predictions of that model

        # concat all features
        predictions = np.hstack(predictions)
        if self.verbose: print('Loaded predictions. Shape : ', predictions.shape)
        return predictions

    def _fitTransformBaseModels(self):
        return self.transformBaseModels()

    def _fitBlendingModel(self, X_blend, y):
        self.blending_model_cv = []

        for model_id, blend_model in enumerate(self.blending_models):
            if self.verbose:
                model_name = "%s" % blend_model.__repr__()
                print('Fitting Blending Model:\n%s' % model_name)

            scores = []
            skf = StratifiedKFold(self.n_folds, shuffle=True, random_state=1000)

            for j, (train_idx, test_idx) in enumerate(skf.split(X_blend, y)):
                if self.verbose:
                    print('Fold %d' % (j + 1))

                X_train, y_train = X_blend[train_idx], y[train_idx]
                X_test, y_test = X_blend[test_idx], y[test_idx]

                model = copy(blend_model)

                if isinstance(model, keras_models.Model) or isinstance(model, keras_models.Sequential):
                    model_path = 'stack_model/keras_model_%d_cv_%d' % (model_id + 1, j + 1) + '.h5'
                    checkpoint = ModelCheckpoint(model_path,
                                                 monitor='val_fbeta_score', verbose=1,
                                                 save_best_only=True, save_weights_only=True,
                                                 mode='max')

                    reduce_lr = ReduceLROnPlateau(monitor='val_fbeta_score', patience=5, mode='max',
                                                  factor=0.8, cooldown=5, min_lr=1e-6, verbose=2)

                    y_train_categorical = to_categorical(y_train, 3)
                    y_test_categorical = to_categorical(y_test, 3)

                    model.fit(X_train, y_train_categorical, batch_size=128, nb_epoch=50, callbacks=[checkpoint, reduce_lr],
                              validation_data=(X_test, y_test_categorical))

                    model.load_weights(model_path)

                    preds = model.predict(X_test, batch_size=128)
                    preds = np.argmax(preds, axis=1)

                    score = f1_score(y_test, preds, average='micro')
                    scores.append(score)
                    print('Keras Model %d - CV %d Score : %0.3f' % (model_id + 1, j + 1, score))

                else:
                    model_path = 'stack_model/sklearn_model_%d_cv_%d' % (model_id + 1, j + 1) + '.pkl'
                    model.fit(X_train, y_train)

                    preds = get_predictions(model, X_test)
                    preds = np.argmax(preds, axis=1)

                    score = f1_score(y_test, preds, average='micro')
                    scores.append(score)
                    print('SKLearn Model %d - CV %d Score : %0.3f' % (model_id + 1, j + 1, score))

                    joblib.dump(model, model_path)

                # add trained model to list of CV'd models
                self.blending_model_cv.append(model)

            print('Average F1 score of model : ', sum(scores) / len(scores))

    def _transformBlendingModel(self, X_blend):
        # make predictions from averaged models
        cv_predictions = None
        n_models = len(self.blending_model_cv)

        for i, model in enumerate(self.blending_model_cv):
            if self.verbose: print('Getting predictions from blending model %s (Classifier id %d)' %
                                   (model.__class__.__name__, i + 1))
            cv_predictions = None
            model_predictions = get_predictions(model, X_blend)

            if cv_predictions is None:
                cv_predictions = np.zeros((n_models, X_blend.shape[0], model_predictions.shape[1]))

            cv_predictions[i,:,:] = model_predictions

        # perform model averaging to get predictions
        predictions = cv_predictions.mean(0)
        return predictions

if __name__ == '__main__':
    model = StackedGeneralizer()

    X_blend = model.transformBaseModels()
    print(X_blend.shape)
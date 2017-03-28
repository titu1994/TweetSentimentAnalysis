import numpy as np
import glob
import joblib
import multiprocessing
import time

import xgboost as xgb

from stacked_generalization import StackedGeneralizer
from sklearn.linear_model import LogisticRegression

import sklearn_utils as sk_utils

from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.models import Model

from sklearn_utils import model_dirs as sklearn_dirs
from keras_utils import model_dirs as keras_dirs

def keras_model_gen():
    X_blend = StackedGeneralizer().transformBaseModels()
    nb_models = X_blend.shape[1]
    # train a fully convolution network
    ip = Input(shape=(nb_models,))
    x = Dense(1024)(ip)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    preds = Dense(3, activation='softmax')(x)

    model = Model(ip, preds)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'fbeta_score'])
    return model


def stack_model_gen(n_folds=10):
    model1 = LogisticRegression(C=1)
    model2 = LogisticRegression(C=0.1)
    model3 = LogisticRegression(C=0.01)
    model4 = LogisticRegression(C=0.001)


    models = [model1, model2, model3, model4, ]

    model = StackedGeneralizer(blending_models=models, n_folds=n_folds, check_dirs=None)
    return model


def fit(model_fn):
    data, labels = sk_utils.prepare_data() # train

    model = stack_model_gen()

    t1 = time.time()
    model.fit(None, labels)
    t2 = time.time()

    print('\nClassifier stack training time : %0.3f seconds.' % (t2 - t1))

    _, test_labels = sk_utils.prepare_data(mode='test')

    print('Begin testing stacked classifier ')

    t1 = time.time()
    preds = model.predict(pred_directory='test/*/', X_indices=None)
    t2 = time.time()

    print('Classifier stack finished predicting in %0.3f seconds.' % (t2 - t1))

    print('\nEvaluationg stack model')
    sk_utils.evaluate(test_labels, preds)
    print()

    joblib.dump(model, '%s.pkl' % (model_fn))


def write_predictions(model_fn='stack_model/stack-model', dataset='full'):
    basepath = model_fn
    path = basepath + ".pkl"

    data, labels = sk_utils.prepare_data(mode='test', dataset=dataset)
    model = joblib.load(path) # type: StackedGeneralizer

    if dataset == 'full':
        pred_dir = 'test/*/'
    elif dataset == 'obama':
        pred_dir = 'obama/*/'
    else:
        pred_dir = 'romney/*/'

    preds_proba = model.predict_proba(pred_dir)
    preds = np.argmax(preds_proba, axis=1)

    sk_utils.evaluate(labels, preds)

    np.save("stack_model/stack_predictions-%s.npy" % (dataset), preds_proba)

if __name__ == '__main__':
    fit(model_fn='stack_model/stack-model')

    #write_predictions()
    #write_predictions(dataset='obama')
    #write_predictions(dataset='romney')


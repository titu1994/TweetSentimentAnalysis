import numpy as np
import glob
import joblib
import multiprocessing
import time

import xgboost as xgb

from stacked_generalization import StackedGeneralizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

import sklearn_utils as sk_utils
import keras_utils as k_utils

from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.models import Model


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


def stack_model_gen(C=1., max_depth=6, subsample=0.95, n_folds=10):
    model1 = LogisticRegression(C=C)
    model2 = keras_model_gen()
    model3 = xgb.XGBClassifier(max_depth=max_depth, learning_rate=0.01,
                               n_estimators=500, objective='multi:softmax',
                               subsample=subsample, seed=1000)

    models = [model1, ]

    model = StackedGeneralizer(blending_models=models, n_folds=n_folds)
    return model


def fit(model_fn, use_full_data=False, seed=1000):
    data, labels = sk_utils.prepare_data(use_full_data)

    model = stack_model_gen()

    t1 = time.time()
    model.fit(None, labels)
    t2 = time.time()

    print('\nClassifier stack training time : %0.3f seconds.' % (t2 - t1))

    print('Begin testing stacked classifier ')

    t1 = time.time()
    preds = model.predict(pred_directory='models/*/', X_indices=None)
    t2 = time.time()

    print('Classifier stack finished predicting in %0.3f seconds.' % (t2 - t1))

    f1score = f1_score(labels, preds, average='micro')

    print('\nF1 Scores of Stack Estimator: %0.4f' % (f1score))

    joblib.dump(model, '%s.pkl' % (model_fn))

def write_predictions(model_dir='stack/'):
    basepath = 'models/' + model_dir
    path = basepath + "*.pkl"

    data, labels = sk_utils.prepare_data()
    files = glob.glob(path)

    X_blend = StackedGeneralizer().transformBaseModels()
    nb_models = X_blend.shape[1]

    nb_models = len(files)

    model_predictions = np.zeros((nb_models, data.shape[0], 3))

    for i, fn in enumerate(files):
        model = joblib.load(fn) # type: StackedGeneralizer

        model_predictions[i, :, :] = model.predict_proba('models/*/')
        print('Finished prediction for model %d' % (i + 1))

    np.save(basepath + "stack_predictions.npy", model_predictions)

if __name__ == '__main__':
    fit(model_fn='stack_model/stack-model')

    # model = joblib.load('stack_model/stack-model.pkl') #type: StackedGeneralizer
    # data, labels = sk_utils.prepare_data(False)
    #
    # preds = model.predict('models/*/')
    # score = f1_score(labels, preds, average='micro')
    #
    # print('F1 score : ', score)
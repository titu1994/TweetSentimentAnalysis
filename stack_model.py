import numpy as np
import glob
import joblib
import multiprocessing
import time

import xgboost as xgb

from stacked_generalization import StackedGeneralizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

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


def fit(model_fn, seed=1000):
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
    model.evaluate(test_labels, preds)
    print()

    f1score = f1_score(test_labels, preds, average='micro')

    print('\nF1 Scores of Stack Estimator: %0.4f' % (f1score))

    joblib.dump(model, '%s.pkl' % (model_fn))


def write_predictions(model_fn='stack_model/stack-model', model_dir='stack/'):
    basepath = model_fn + model_dir
    path = basepath + "*.pkl"

    data, labels = sk_utils.prepare_data()
    files = glob.glob(path)

    X_blend = StackedGeneralizer().transformBaseModels()
    nb_models = X_blend.shape[1]

    #nb_models = len(files)

    model_predictions = np.zeros((nb_models, data.shape[0], 3))

    for i, fn in enumerate(files):
        model = joblib.load(fn) # type: StackedGeneralizer

        model_predictions[i, :, :] = model.predict_proba('data/*/')
        print('Finished prediction for model %d' % (i + 1))

    np.save(basepath + "stack_predictions.npy", model_predictions)

if __name__ == '__main__':
    fit(model_fn='stack_model/stack-model')

    # model = joblib.load('stack_model/stack-model.pkl') #type: StackedGeneralizer
    # data, labels = sk_utils.prepare_data()
    #
    # preds = model.predict('models/*/', X_indices=None)
    # score = f1_score(labels, preds, average='micro')
    #
    # print('Evaluating Model : ')
    # print('\t\t\t\t\t\t\tClasses')
    # print('\t\t\t\t-1\t\t\t\t0\t\t\t\t+1\n')
    # model.evaluate(labels, preds)
    #
    # print('\nF1 score : ', score)


import pandas as pd
import numpy as np
import joblib
import time
import os
import re
import glob
import ast
import pickle

import xgboost as xgb

np.random.seed(1000)

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, accuracy_score, confusion_matrix

train_obama_path = "data/obama_csv.csv"
train_romney_path = "data/romney_csv.csv"

test_obama_path = "data/obama_csv_test.csv"
test_romney_path = "data/romney_csv_test.csv"

if not os.path.exists('models/'):
    os.makedirs('models/')

if not os.path.exists('test/'):
    os.makedirs('test/')

if not os.path.exists('stack_model/'):
    os.makedirs('stack_model/')

if not os.path.exists('vote_model/'):
    os.makedirs('vote_model/')

subdirs = ['conv/', 'lstm/', 'bidirectional_lstm/', 'n_conv/',  'xgboost/', 'mnb/', 'svm/', 'nbsvm/', 'logistic/',
           'sgd/', 'voting/', 'ridge/', 'multiplicative_lstm/']

for sub in subdirs:
    for base in ['models/', 'test/', 'obama/', 'romney/']:
        path = base + sub

        if not os.path.exists(path):
            os.makedirs(path)

# Note, XGBoost requires data in DMatrix format, MUST be the last model in model_dirs!
model_dirs = ['logistic/', 'mnb/', 'nbsvm/', 'svm/', 'sgd/', 'ridge/', ]
model_dirs.append('xgboost/')


def _get_predictions(model, X):
    if hasattr(model, 'predict_proba'):  # Normal SKLearn classifiers
        pred = model.predict_proba(X)
    elif hasattr(model, '_predict_proba_lr'):  # SVMs
        pred = model._predict_proba_lr(X)
    else:
        pred = model.predict(X)

    if len(pred.shape) == 1:  # for 1-d ouputs
        pred = pred[:, None]

    return pred


def load_obama(mode='train'):
    if mode == 'train':
        obama_df = pd.read_csv(train_obama_path, sep='\t', encoding='latin1')
    else:
        obama_df = pd.read_csv(test_obama_path, sep='\t', encoding='latin1')
    # Remove rows who have no class label attached, can hand label later
    obama_df = obama_df[pd.notnull(obama_df['label'])]
    obama_df['label'] = obama_df['label'].astype(np.int)

    texts = []  # list of text samples
    labels_index = {-1: 0,
                    0: 1,
                    1: 2}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids

    obama_df = obama_df[obama_df['label'] != 2]  # drop all rows with class = 2

    nb_rows = len(obama_df)
    for i in range(nb_rows):
        row = obama_df.iloc[i]
        texts.append(str(row['tweet']))
        labels.append(labels_index[int(row['label'])])

    texts = np.asarray(texts)
    labels = np.asarray(labels)

    return texts, labels, labels_index


def load_romney(mode='train'):
    if mode == 'train':
        romney_df = pd.read_csv(train_romney_path, sep='\t', encoding='latin1')
    else:
        romney_df = pd.read_csv(test_romney_path, sep='\t', encoding='latin1')
    # Remove rows who have no class label attached, can hand label later
    romney_df = romney_df[pd.notnull(romney_df['label'])]
    romney_df['label'] = romney_df['label'].astype(np.int)

    texts = []  # list of text samples
    labels_index = {-1: 0,
                    0: 1,
                    1: 2}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids

    romney_df = romney_df[romney_df['label'] != 2]  # drop all rows with class = 2

    nb_rows = len(romney_df)
    for i in range(nb_rows):
        row = romney_df.iloc[i]
        texts.append(str(row['tweet']))
        labels.append(labels_index[int(row['label'])])

    texts = np.asarray(texts)
    labels = np.asarray(labels)

    return texts, labels, labels_index


def load_both(mode='train'):
    if mode == 'train':
        obama_df = pd.read_csv(train_obama_path, sep='\t', encoding='latin1')
    else:
        obama_df = pd.read_csv(test_obama_path, sep='\t', encoding='latin1')
    # Remove rows who have no class label attached, can hand label later
    obama_df = obama_df[pd.notnull(obama_df['label'])]
    obama_df['label'] = obama_df['label'].astype(np.int)

    if mode == 'train':
        romney_df = pd.read_csv(train_romney_path, sep='\t', encoding='latin1')
    else:
        romney_df = pd.read_csv(test_romney_path, sep='\t', encoding='latin1')
    # Remove rows who have no class label attached, can hand label later
    romney_df = romney_df[pd.notnull(romney_df['label'])]
    romney_df['label'] = romney_df['label'].astype(np.int)

    texts = []  # list of text samples
    labels_index = {-1: 0,
                    0: 1,
                    1: 2}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids

    obama_df = obama_df[obama_df['label'] != 2]  # drop all rows with class = 2
    romney_df = romney_df[romney_df['label'] != 2]  # drop all rows with class = 2

    nb_rows = len(obama_df)
    for i in range(nb_rows):
        row = obama_df.iloc[i]
        texts.append(str(row['tweet']))
        labels.append(labels_index[int(row['label'])])

    nb_rows = len(romney_df)
    for i in range(nb_rows):
        row = romney_df.iloc[i]
        texts.append(str(row['tweet']))
        labels.append(labels_index[int(row['label'])])

    texts = np.asarray(texts)
    labels = np.asarray(labels)

    return texts, labels, labels_index


def tokenize(texts):
    if os.path.exists('data/vectorizer.pkl'):
        with open('data/vectorizer.pkl', 'rb') as f:
            tfidf_vect = pickle.load(f)
            x_counts = tfidf_vect.transform(texts)
    else:
        tfidf_vect = CountVectorizer(ngram_range=(1, 2))
        x_counts = tfidf_vect.fit_transform(texts)

        with open('data/vectorizer.pkl', 'wb') as f:
            pickle.dump(tfidf_vect, f)

    print('Shape of tokenizer counts : ', x_counts.shape)
    return x_counts


def tfidf(x_counts):
    if os.path.exists('data/tfidf.pkl'):
        with open('data/tfidf.pkl', 'rb') as f:
            transformer = pickle.load(f)
            x_tfidf = transformer.transform(x_counts)
    else:
        transformer = TfidfTransformer()
        x_tfidf = transformer.fit_transform(x_counts)

        with open('data/tfidf.pkl', 'wb') as f:
            pickle.dump(transformer, f)

    return x_tfidf


def train_sklearn_model_cv(model_gen, model_fn, k_folds=3, seed=1000):
    data, labels = prepare_data()

    skf = StratifiedKFold(k_folds, shuffle=True, random_state=seed)

    fbeta_scores = []

    for i, (train_idx, test_idx) in enumerate(skf.split(data, labels)):
        x_train, y_train = data[train_idx, :], labels[train_idx]
        x_test, y_test = data[test_idx, :], labels[test_idx]

        print('\nBegin training classifier %d' % (i + 1), 'Training samples : ', x_train.shape[0],
              'Testing samples : ', x_test.shape[0])

        model = model_gen()

        t1 = time.time()
        try:
            model.fit(x_train, y_train)
        except TypeError:
            # Model does not support sparce matrix input, convert to dense matrix input
            x_train = x_train.toarray()
            model.fit(x_train, y_train)

            # dense matrix input is very large, delete to preserve memory
            del (x_train)

        t2 = time.time()

        print('Classifier %d training time : %0.3f seconds.' % (i + 1, t2 - t1))

        print('Begin testing classifier %d' % (i + 1))

        t1 = time.time()
        try:
            preds = model.predict(x_test)
        except TypeError:
            # Model does not support sparce matrix input, convert to dense matrix input
            x_test = x_test.toarray()
            preds = model.predict(x_test)

            # dense matrix input is very large, delete to preserve memory
            del (x_test)

        t2 = time.time()

        print('Classifier %d finished predicting in %0.3f seconds.' % (i + 1, t2 - t1))

        f1score = f1_score(y_test, preds, average='micro')
        fbeta_scores.append(f1score)

        print('\nF1 Scores of Estimator %d: %0.4f' % (i + 1, f1score))

        joblib.dump(model, 'models/%s-cv-%d.pkl' % (model_fn, i + 1))

        del model

    print("\nAverage fbeta score : ", sum(fbeta_scores) / len(fbeta_scores))

    with open('models/%s-scores.txt' % (model_fn), 'w') as f:
        f.write(str(fbeta_scores))

def prepare_data(mode='train', dataset='full', verbose=True):
    assert dataset in ['full', 'obama', 'romney']

    if verbose: print('Loading %s data' % mode)

    if dataset == 'full':
        texts, labels, label_map = load_both(mode)
    elif dataset == 'obama':
        texts, labels, label_map = load_obama(mode)
    else:
        texts, labels, label_map = load_romney(mode)

    if verbose: print('Tokenizing texts')
    x_counts = tokenize(texts)
    if verbose: print('Finished tokenizing texts')
    data = tfidf(x_counts)
    if verbose:
        print('Finished computing TF-IDF')
        print('-' * 80)
    return data, labels


def load_trained_sklearn_models(model_dir_base='models/', normalize_weights=False, verbose=True):
    name_clfs = []
    clf_scores = []
    index = 0

    for model_dir in model_dirs:
        path = model_dir_base + model_dir + '*.pkl'
        weights_path = model_dir_base + model_dir + '*.txt'

        weight_path = glob.glob(weights_path)
        if verbose: print('Loading weight file [0]:', weight_path)

        with open(weight_path[0], 'r') as f:
            clf_weight_data = ast.literal_eval(f.readline())

        fns = glob.glob(path)
        cv_ids = []
        for i in range(len(fns)):
            fn = fns[i]
            cv_id = re.search(r'\d+', fn).group()
            cv_ids.append(int(cv_id))

        clf_weight_data = [clf_weight_data[i - 1] for i in cv_ids]
        clf_scores.extend(clf_weight_data)

        for fn in fns:
            model = joblib.load(fn)
            name_clfs.append(('%s_%d' % (model.__class__.__name__, index + 1), model))
            if verbose: print('Added model - %s as %s_%d' % (fn, model.__class__.__name__, index + 1))

            index += 1

        if verbose: print()

    if normalize_weights:
        weight_sum = np.sum(np.asarray(clf_scores, dtype=np.float32))
        weights = [w / weight_sum for w in clf_scores]
        clf_scores = weights

    return (name_clfs, clf_scores)


def get_sklearn_scores(normalize_scores=False):
    clf_scores = []

    for m, model_dir in enumerate(model_dirs):
        weights_path = 'models/' + model_dir + '*.txt'

        weight_path = glob.glob(weights_path)
        print('Loading score file [0]:', weight_path)

        with open(weight_path[0], 'r') as f:
            clf_weight_data = ast.literal_eval(f.readline())

        clf_scores.extend(clf_weight_data)

    if normalize_scores:
        weight_sum = np.sum(np.asarray(clf_scores, dtype=np.float32))
        weights = [w / weight_sum for w in clf_scores]
        clf_scores = weights

    return clf_scores


def get_predictions_sklearn_models(data, save_path='test/', normalize_weights=False):
    model_preds = []
    clf_scores = []

    for m, model_dir in enumerate(model_dirs):
        path = 'models/' + model_dir + '*.pkl'
        weights_path = 'models/' + model_dir + '*.txt'

        weight_path = glob.glob(weights_path)
        print('Loading weight file [0]:', weight_path)

        with open(weight_path[0], 'r') as f:
            clf_weight_data = ast.literal_eval(f.readline())

        fns = glob.glob(path)
        cv_ids = []
        for i in range(len(fns)):
            fn = fns[i]
            cv_id = re.search(r'\d+', fn).group()
            cv_ids.append(int(cv_id))

        clf_weight_data = [clf_weight_data[i - 1] for i in cv_ids]
        clf_scores.extend(clf_weight_data)

        temp_preds = np.zeros((len(cv_ids), data.shape[0], 3))

        xgb_data = xgb.DMatrix(data)

        for j, fn in enumerate(fns):
            model = joblib.load(fn)
            if not 'xgb' in fn:
                preds = _get_predictions(model, data)
            else:
                preds = _get_predictions(model, xgb_data)

            temp_preds[j, :, :] = preds

            print('Got predictions for model - %s' % (fn))

        model_preds.append(temp_preds) # temp_preds.mean(axis=0)

        preds_save_path = save_path + model_dir + os.path.splitext(os.path.basename(weight_path[0]))[0] + '.npy'
        preds = temp_preds

        np.save(preds_save_path, preds)
        print('Saved predictions for %s in %s' % (model_dir[:-1], preds_save_path))

        print()

    if normalize_weights:
        weight_sum = np.sum(np.asarray(clf_scores, dtype=np.float32))
        weights = [w / weight_sum for w in clf_scores]
        clf_scores = weights

    return (model_preds, clf_scores)


def evaluate_sklearn_model(model_dir, dataset='full'):
    data, labels = prepare_data(mode='test', dataset=dataset)

    path = 'models/' + model_dir + '*.pkl'
    fns = glob.glob(path)
    temp_preds = np.zeros((len(fns), data.shape[0], 3))

    xgb_data = xgb.DMatrix(data)
    class_name = ""

    for j, fn in enumerate(fns):
        model = joblib.load(fn)
        class_name = model.__class__.__name__

        if not 'xgb' in fn:
            preds = _get_predictions(model, data)
        else:
            preds = _get_predictions(model, xgb_data)

        temp_preds[j, :, :] = preds

        #print('Got predictions for model - %s' % (fn))

    preds = temp_preds.mean(axis=0)
    preds = np.argmax(preds, axis=1)

    print('Evaluating Model %s' % class_name)
    print()
    evaluate(labels, preds)


def evaluate(y_true, y_pred):
    f1score = f1_score(y_true, y_pred, labels=[0, 2], average='micro')
    print('F1 score (+/- classes) : ', f1score)
    print()
    print(classification_report(y_true, y_pred, labels=[0, 2]))
    print('Confusion Matrix (+/- classes) :\n')
    print('\tClasses')
    print('-1\t\t+1')
    print(confusion_matrix(y_true, y_pred, labels=[0, 2]))
    print()
    print('Accuracy Score (all 3 classes) : ', accuracy_score(y_true, y_pred))
    print()


if __name__ == '__main__':
    #test_data, labels = prepare_data(mode='test')
    #get_predictions_sklearn_models(test_data, save_path='obama/')

    obama_data, obama_labels = prepare_data(mode='test', dataset='obama')
    get_predictions_sklearn_models(obama_data, save_path='obama/')

    romney_data, romney_labels = prepare_data(mode='test', dataset='romney')
    get_predictions_sklearn_models(romney_data, save_path='romney/')
    pass

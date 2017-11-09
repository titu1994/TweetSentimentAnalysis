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


# create the necessary workspace directories for saving files and weights
if not os.path.exists('models/'):
    os.makedirs('models/')

if not os.path.exists('test/'):
    os.makedirs('test/')

if not os.path.exists('stack_model/'):
    os.makedirs('stack_model/')

if not os.path.exists('vote_model/'):
    os.makedirs('vote_model/')

# list of all ML models directories and their paths to the directories
# if you add more models, add their paths here to create their subdirectories
subdirs = ['conv/', 'lstm/', 'bidirectional_lstm/', 'n_conv/',  'xgboost/', 'mnb/', 'svm/', 'nbsvm/', 'logistic/',
           'sgd/', 'voting/', 'ridge/', 'multiplicative_lstm/']

# create all the directories and subdirectories required
for sub in subdirs:
    for base in ['models/', 'test/', 'obama/', 'romney/']:
        path = base + sub

        if not os.path.exists(path):
            os.makedirs(path)

# Note, XGBoost requires data in DMatrix format, MUST be the last model in model_dirs!
model_dirs = ['logistic/', 'mnb/', 'nbsvm/', 'svm/', 'sgd/', 'ridge/', ]
model_dirs.append('xgboost/')


def _get_predictions(model, X):
    '''
    Wrapper function to get predictions transparently from either
    SKLearn, XGBoost, Ensemble or Keras models.

    Args:
        model: Model for prediction
        X: input data in correct format for prediction

    Returns:
        predictions
    '''
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
    '''
    Loads the Obama dataset

    Args:
        mode: decides whether to load train or test set

    Returns:
        raw text list, labels and label indices
    '''
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
    '''
    Loads the Romney dataset

    Args:
        mode: decides whether to load train or test set

    Returns:
        raw text list, labels and label indices
    '''

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
    '''
    Loads both Obama and Romney datasets for Joint Training

    Args:
        mode: decides whether to load train or test set

    Returns:
        raw text list, labels and label indices
    '''
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
    '''
    For SKLearn models / XGBoost / Ensemble, use CountVectorizer to generate
    n-gram vectorized texts efficiently.

    Args:
        texts: input text sentences list

    Returns:
        the n-gram text
    '''
    if os.path.exists('data/vectorizer.pkl'):
        with open('data/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
            x_counts = vectorizer.transform(texts)
    else:
        vectorizer = CountVectorizer(ngram_range=(1, 2))
        x_counts = vectorizer.fit_transform(texts)

        with open('data/vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)

    print('Shape of tokenizer counts : ', x_counts.shape)
    return x_counts


def tfidf(x_counts):
    '''
    Perform TF-IDF transform to normalize the dataset

    Args:
        x_counts: the n-gram tokenized sentences

    Returns:
        the TF-IDF transformed dataset
    '''
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
    '''
    Standard training Stratifierd Cross Validated training for all
    SKLearn / XGBoost / Ensemble models.

    Args:
        model_gen: a function which generates a SKLearn / XGBoost / Ensemble model
        model_fn: a filename for the model for serialiation of models
        k_folds: number of folds
        seed: random seed value for Stratified K Folds

    Returns:

    '''
    data, labels = prepare_data()  # prepare the full training dataset for Joint Training

    skf = StratifiedKFold(k_folds, shuffle=True, random_state=seed)

    fbeta_scores = []

    for i, (train_idx, test_idx) in enumerate(skf.split(data, labels)):
        x_train, y_train = data[train_idx, :], labels[train_idx]  # obtain train fold
        x_test, y_test = data[test_idx, :], labels[test_idx]  # obtain test fold

        print('\nBegin training classifier %d' % (i + 1), 'Training samples : ', x_train.shape[0],
              'Testing samples : ', x_test.shape[0])

        model = model_gen()  # generate a model from the model generator

        t1 = time.time()
        try:
            model.fit(x_train, y_train)  # attempt to fit the dataset using sparse numpy arrays
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
            preds = model.predict(x_test)  # attempt to obtain predictions using sparce numpy arrays
        except TypeError:
            # Model does not support sparce matrix input, convert to dense matrix input
            x_test = x_test.toarray()
            preds = model.predict(x_test)

            # dense matrix input is very large, delete to preserve memory
            del (x_test)

        t2 = time.time()

        print('Classifier %d finished predicting in %0.3f seconds.' % (i + 1, t2 - t1))

        f1score = f1_score(y_test, preds, average='micro')  # compute micro f1 score since over 3 classes
        fbeta_scores.append(f1score)

        print('\nF1 Scores of Estimator %d: %0.4f' % (i + 1, f1score))

        joblib.dump(model, 'models/%s-cv-%d.pkl' % (model_fn, i + 1))  # serialize the trained model

        del model  # delete the trained model from CPU memory

    print("\nAverage fbeta score : ", sum(fbeta_scores) / len(fbeta_scores))  # compute average f1 score over all folds

    with open('models/%s-scores.txt' % (model_fn), 'w') as f:
        f.write(str(fbeta_scores))  # serialize the f1 scores

def prepare_data(mode='train', dataset='full', verbose=True):
    '''
    Utility function to load a given dataset with a certain mode

    Args:
        mode: can be 'train' or 'test'
        dataset: can be 'full', 'obama' or 'romney'
        verbose: set to True to obtain information of loaded dataset

    Returns:
        tokenized tf-idf transformed input text and corresponding labels
    '''
    assert dataset in ['full', 'obama', 'romney']

    if verbose: print('Loading %s data' % mode)

    if dataset == 'full':
        texts, labels, label_map = load_both(mode)
    elif dataset == 'obama':
        texts, labels, label_map = load_obama(mode)
    else:
        texts, labels, label_map = load_romney(mode)

    if verbose: print('Tokenizing texts')
    x_counts = tokenize(texts)  # tokenize the loaded texts

    if verbose: print('Finished tokenizing texts')
    data = tfidf(x_counts)  # transform the loaded texts

    if verbose:
        print('Finished computing TF-IDF')
        print('-' * 80)

    return data, labels


def load_trained_sklearn_models(model_dir_base='models/', normalize_weights=False, verbose=True):
    '''
    Utility function to de-serialize a trained SKLearn models
    and obtain their predictions.

    Args:
        model_dir_base: The path to the serialized folds of the model
        normalize_weights: whether to perform weight normalization
        verbose: set to True to obtain information about prediction status

    Returns:
        a tuple of the name of the classifier/s and its corresponding f1 score/s
    '''
    name_clfs = []
    clf_scores = []
    index = 0

    for model_dir in model_dirs:  # for all the SKLearn models that are have been serialized
        path = model_dir_base + model_dir + '*.pkl'
        weights_path = model_dir_base + model_dir + '*.txt'

        weight_path = glob.glob(weights_path)  # obtain all the serialized folds scores
        if verbose: print('Loading weight file [0]:', weight_path)

        with open(weight_path[0], 'r') as f:
            clf_weight_data = ast.literal_eval(f.readline())  # load the serialized scores list into memory

        # compute the cv scores for each fold
        fns = glob.glob(path)  # get the paths to all of the serialized models
        cv_ids = []
        for i in range(len(fns)):
            fn = fns[i]  # get the i-th score path
            cv_id = re.search(r'\d+', fn).group()  # get the corresponding index for the f1 score in the list
            cv_ids.append(int(cv_id))  # store index

        # obtain the actual f1 scores from the list in correct order for their corresponding models
        clf_weight_data = [clf_weight_data[i - 1] for i in cv_ids]
        clf_scores.extend(clf_weight_data)

        # get the corresponding name of the classifier for the above f1 scores
        for fn in fns:
            model = joblib.load(fn)
            name_clfs.append(('%s_%d' % (model.__class__.__name__, index + 1), model))
            if verbose: print('Added model - %s as %s_%d' % (fn, model.__class__.__name__, index + 1))

            index += 1

        if verbose: print()

    # normalize weights if required
    if normalize_weights:
        weight_sum = np.sum(np.asarray(clf_scores, dtype=np.float32))
        weights = [w / weight_sum for w in clf_scores]
        clf_scores = weights

    return (name_clfs, clf_scores)


def get_sklearn_scores(normalize_scores=False):
    '''
    Utility function to get a list of all the scores for all
    SKLearn / XGBoost / Ensemble models.

    Args:
        normalize_scores: whether to normalize the scores or not

    Returns:
        a list of all loaded scores
    '''
    clf_scores = []

    for m, model_dir in enumerate(model_dirs):
        weights_path = 'models/' + model_dir + '*.txt'  # for each model path

        weight_path = glob.glob(weights_path)  # get the score list of the corresponding model
        print('Loading score file [0]:', weight_path)

        with open(weight_path[0], 'r') as f:
            clf_weight_data = ast.literal_eval(f.readline())  # parse the score list into a python list

        clf_scores.extend(clf_weight_data)  # append it to the overall score list

    # normalize scores if needed
    if normalize_scores:
        weight_sum = np.sum(np.asarray(clf_scores, dtype=np.float32))
        weights = [w / weight_sum for w in clf_scores]
        clf_scores = weights

    return clf_scores


def get_predictions_sklearn_models(data, save_path='test/', normalize_weights=False):
    '''
    Uniform interface to extract predictions from SKLearn / XGBoost /
    Ensemble models.

    Args:
        data: input data
        save_path: path to save the predictions
        normalize_weights: whether to normalize the weights of the predictions

    Returns:
        the models predictions and the corresponding f1 score of that model (during CV training)
    '''
    model_preds = []
    clf_scores = []

    for m, model_dir in enumerate(model_dirs):  # for all models
        path = 'models/' + model_dir + '*.pkl'
        weights_path = 'models/' + model_dir + '*.txt'

        weight_path = glob.glob(weights_path)  # obtain all the serialized folds scores
        print('Loading weight file [0]:', weight_path)

        with open(weight_path[0], 'r') as f:
            clf_weight_data = ast.literal_eval(f.readline())   # parse the score list into a python list

        # compute the cv scores for each fold
        fns = glob.glob(path)  # get the paths to all of the serialized models
        cv_ids = []
        for i in range(len(fns)):
            fn = fns[i]  # get the i-th score path
            cv_id = re.search(r'\d+', fn).group()  # get the corresponding index for the f1 score in the list
            cv_ids.append(int(cv_id))  # store index

        # obtain the actual f1 scores from the list in correct order for their corresponding models
        clf_weight_data = [clf_weight_data[i - 1] for i in cv_ids]
        clf_scores.extend(clf_weight_data)

        # create a buffer to store all of the predictions from each fold for each model
        temp_preds = np.zeros((len(cv_ids), data.shape[0], 3))  # final 3 is for the 3 classes

        for j, fn in enumerate(fns):
            model = joblib.load(fn)  # deserialize the SKLearn / XGboost model

            if not 'xgb' in fn:  # if SKLearn model
                preds = _get_predictions(model, data)  # get predictions
            else:  # is a XGBoost model
                # prepare the input data in a format compatible to XGBoost
                xgb_data = xgb.DMatrix(data)
                preds = _get_predictions(model, xgb_data)  # get predictions

            temp_preds[j, :, :] = preds  # for the jth fold, preserve the predictions

            print('Got predictions for model - %s' % (fn))

        model_preds.append(temp_preds)  # save the predictions in a list per model

        preds_save_path = save_path + model_dir + os.path.splitext(os.path.basename(weight_path[0]))[0] + '.npy'
        preds = temp_preds

        np.save(preds_save_path, preds)  # save the prediction matrix over all folds into a numpy array
        print('Saved predictions for %s in %s' % (model_dir[:-1], preds_save_path))

        print()

    # normalize f1 scores if needed
    if normalize_weights:
        weight_sum = np.sum(np.asarray(clf_scores, dtype=np.float32))
        weights = [w / weight_sum for w in clf_scores]
        clf_scores = weights

    return (model_preds, clf_scores)


def evaluate_sklearn_model(model_dir, dataset='full'):
    '''
    Utility function to evaluate the performance of SKLearn / XGBoost / Ensemble models

    Args:
        model_dir: path to the model directory
    '''
    data, labels = prepare_data(mode='test', dataset=dataset)

    path = 'models/' + model_dir + '*.pkl'  # path to model directory
    fns = glob.glob(path)  # get all fold names

    # create a buffer to store all of the predictions from each fold for each model
    temp_preds = np.zeros((len(fns), data.shape[0], 3))

    class_name = ""
    for j, fn in enumerate(fns):
        model = joblib.load(fn)  # deserialize the model
        class_name = model.__class__.__name__  # get the model name

        if not 'xgb' in fn:  # if SKLearn model
            preds = _get_predictions(model, data)
        else:  # if XGBoost model
            xgb_data = xgb.DMatrix(data)
            preds = _get_predictions(model, xgb_data)

        temp_preds[j, :, :] = preds  # for the jth fold, preserve the predictions

    preds = temp_preds.mean(axis=0)  # calculate the average prediction over all folds
    preds = np.argmax(preds, axis=1)  # compute the maximum probability class

    print('Evaluating Model %s' % class_name)
    print()
    evaluate(labels, preds)  # evaluate the model


def evaluate(y_true, y_pred):
    '''
    Helper method for evaluating any model

    Args:
        y_true: true class labels
        y_pred: predicted class labels
    '''
    # compute f1 score over only negative and positive classes
    f1score = f1_score(y_true, y_pred, labels=[0, 2], average='micro')

    print('F1 score (+/- classes) : ', f1score)
    print()

    # prepare a classification report over the negative and positive classes
    print(classification_report(y_true, y_pred, labels=[0, 2]))

    # prepare the confusion matrix over the negative and positive classes
    print('Confusion Matrix (+/- classes) :\n')
    print('\tClasses')
    print('-1\t\t+1')
    print(confusion_matrix(y_true, y_pred, labels=[0, 2]))
    print()

    # obtain the overall accuracy over all classes
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

import numpy as np
import pandas as pd
import time
import os
import pickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from sklearn.naive_bayes import MultinomialNB

train_obama_path = "data/obama_csv.csv"
train_romney_path = "data/romney_csv.csv"

np.random.seed(1000)


def load_both():
    obama_df = pd.read_csv(train_obama_path, sep='\t', encoding='latin1')
    # Remove rows who have no class label attached, can hand label later
    obama_df = obama_df[pd.notnull(obama_df['label'])]
    obama_df['label'] = obama_df['label'].astype(np.int)

    romney_df = pd.read_csv(train_romney_path, sep='\t', encoding='latin1')
    # Remove rows who have no class label attached, can hand label later
    romney_df = romney_df[pd.notnull(romney_df['label'])]
    romney_df['label'] = romney_df['label'].astype(np.int)

    texts = []  # list of text samples
    labels_index = {-1: 0,
                    0: 1,
                    1: 2}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids

    obama_df_class2 = obama_df[obama_df['label'] == 2]  # keep all rows with class = 2
    romney_df_class2 = romney_df[romney_df['label'] == 2]  # keep all rows with class = 2

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

    texts_class2 = []
    nb_rows = len(obama_df_class2)
    for i in range(nb_rows):
        row = obama_df_class2.iloc[i]
        texts_class2.append(str(row['tweet']))

    nb_rows = len(romney_df_class2)
    for i in range(nb_rows):
        row = romney_df_class2.iloc[i]
        texts_class2.append(str(row['tweet']))

    texts_class2 = np.asarray(texts_class2)

    np.random.seed(1000)
    np.random.shuffle(texts_class2)

    return texts, labels, texts_class2


def tokenize(texts):
    tfidf_vect = CountVectorizer(ngram_range=(1, 2))
    x_counts = tfidf_vect.fit_transform(texts)

    with open('data/em_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vect, f)

    print('Shape of tokenizer counts : ', x_counts.shape)
    return x_counts


def tfidf(x_counts):
    transformer = TfidfTransformer()
    x_tfidf = transformer.fit_transform(x_counts)

    with open('data/em_tfidf.pkl', 'wb') as f:
        pickle.dump(transformer, f)

    return x_tfidf


def expectation_maximization(nb_iterations=5, alpha=0.7, verbose=False):
    print('Loading data')
    texts, labels, texts_class2 = load_both()
    print('Tokenizing texts')

    texts_full = np.concatenate((texts, texts_class2))

    x_counts = tokenize(texts_full)
    print('Finished tokenizing texts')

    data = tfidf(x_counts)
    print('Finished computing TF-IDF')

    train_data = data[:texts.shape[0], ...]
    test_data = data[texts.shape[0]:, ...]

    print('Train shape : ', train_data.shape)
    print('Test shape : ', test_data.shape)

    initial_model = MultinomialNB(alpha=0.1)
    initial_model.fit(train_data, labels)

    predicted_labels = initial_model.predict(test_data) # initial predicted labels
    labels_full = np.concatenate((labels, predicted_labels))

    print('Full train shape : ', data.shape)
    print('Full labels shape : ', labels_full.shape)
    print('-' * 80)

    print('Unique labels : ', np.unique(labels_full))

    for i in range(nb_iterations):
        # copy training data to avoid shuffling the original data
        # this is done as after shuffling we cannot determine the location of class 2 objects
        # after they have been given new labels
        X_train = data.copy()
        Y_train = labels_full.copy()

        x_test = X_train[texts.shape[0]:, ...]

        train_indices = np.random.permutation(X_train.shape[0])
        # shuffle the train rows only
        x_train, y_train = X_train[train_indices], Y_train[train_indices]

        print('\nBegin EM step %d' % (i + 1))

        model = MultinomialNB(alpha=alpha)

        t1 = time.time()
        model.fit(x_train, y_train)
        t2 = time.time()

        print('Classifier %d training time : %0.3f seconds.' % (i + 1, t2 - t1))

        print('Begin predicting new labels for class 2')
        # Predict new labels for class 2
        t1 = time.time()
        preds = model.predict(x_test)
        t2 = time.time()

        # update class 2 labels for next iteration
        labels_full[texts.shape[0]:] = preds
        print('Finished predicting labels for class 2 in %0.3f seconds' % (t2 - t1))

        print('Begin predicting class labels : classifier %d' % (i + 1))
        # Predict new labels for class 2
        t1 = time.time()
        preds = model.predict(x_train)
        t2 = time.time()

        print('Classifier %d finished predicting in %0.3f seconds.' % (i + 1, t2 - t1))

        f1score = f1_score(y_train, preds, average='micro')
        print('\nF1 Scores of classifier %d over full data: %0.4f' % (i + 1, f1score))
        print('*' * 80)

        del (model, x_train, y_train)

    write_new_labels(labels_full, texts.shape[0], verbose)


def write_new_labels(new_labels, starting_index, verbose=False):
    obama_df = pd.read_csv(train_obama_path, sep='\t', encoding='latin1')
    # Remove rows who have no class label attached, can hand label later
    obama_df = obama_df[pd.notnull(obama_df['label'])]
    obama_df['label'] = obama_df['label'].astype(np.int)

    obama_count = len(obama_df)

    romney_df = pd.read_csv(train_romney_path, sep='\t', encoding='latin1')
    # Remove rows who have no class label attached, can hand label later
    romney_df = romney_df[pd.notnull(romney_df['label'])]
    romney_df['label'] = romney_df['label'].astype(np.int)

    total_df = pd.concat([obama_df, romney_df])

    print('Unique Labels', np.unique(new_labels))
    nb_rows = len(total_df)
    for i in range(nb_rows):
        if total_df.iloc[i]['label'] == 2:
            total_df.iloc[i, total_df.columns.get_loc('label')] = new_labels[i] - 1

            if verbose:
                print('Changing label at %d from 2 to %d' % (i + 1, new_labels[i] - 1))

    obama_df = total_df[:obama_count]
    romney_df = total_df[obama_count:]

    obama_df.to_csv('data/full_obama_csv.csv', sep='\t')
    romney_df.to_csv('data/full_romney_csv.csv', sep='\t')

    print('Files saved!')


if __name__ == '__main__':
    expectation_maximization(nb_iterations=10, alpha=1e-3, verbose=False)






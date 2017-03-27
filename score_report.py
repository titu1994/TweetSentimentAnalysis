from sklearn_utils import get_sklearn_scores, load_both, model_dirs as sklearn_model_dirs_
from keras_utils import get_keras_scores, prepare_tokenized_data

clf_scores = get_sklearn_scores(normalize_scores=False)

nb_models = len(clf_scores) // 100 # counting only the sklearn models

model_scores = {}

for i in range(nb_models):
    model_name = sklearn_model_dirs_[i]
    scores = []

    for j in range(100):
        model_index = i * 100 + j
        scores.append(clf_scores[model_index])

    print("Adding score of model :", model_name)
    model_scores[model_name[:-1]] = sum(scores) / len(scores)


MAX_NB_WORDS = 95000
MAX_SEQUENCE_LENGTH = 80

texts, labels, label_map = load_both()
data, word_index = prepare_tokenized_data(texts, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)

clf_scores = get_keras_scores(normalize_scores=False)

conv_scores = clf_scores[:10]
n_conv_scores = clf_scores[10:20]
lstm_scores = clf_scores[20:30]
bi_lstm_scores = clf_scores[30:]

print()

print('Cross Validation Scores')
print(['-'] * 80)

for k, v in model_scores.items():
    print("Model %s : Mean Score : %0.5f" % (k, v))

print()

print('Conv Scores : ', sum(conv_scores) / len(conv_scores))
print('N-Conv Scores : ', sum(n_conv_scores) / len(n_conv_scores))
print('LSTM Scores : ', sum(lstm_scores) / len(lstm_scores))
print('Bi-LSTM Scores : ', sum(bi_lstm_scores) / len(bi_lstm_scores))

print()
print('Stacked Generatlizer score : ')
print(['-'] * 80)


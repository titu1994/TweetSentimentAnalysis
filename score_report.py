from sklearn_utils import load_trained_sklearn_models, load_both
from keras_utils import get_keras_scores, prepare_tokenized_data

name_clfs, clf_scores = load_trained_sklearn_models()

nb_models = len(clf_scores) // 100 # counting only the sklearn models

model_scores = {}

for i in range(nb_models):
    model_name = name_clfs[i * 100][1].__class__.__name__
    scores = []

    for j in range(100):
        model_index = i * 100 + j
        scores.append(clf_scores[model_index])

    print("Adding score of model :", model_name)
    model_scores[model_name] = sum(scores) / len(scores)
#


MAX_NB_WORDS = 95000
MAX_SEQUENCE_LENGTH = 80

texts, labels, label_map = load_both()
data, word_index = prepare_tokenized_data(texts, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)

clf_scores = get_keras_scores(normalize_weights=False)

conv_scores = clf_scores[:10]
n_conv_scores = clf_scores[10:20]
lstm_scores = clf_scores[20:]

print()

for k, v in model_scores.items():
    print("Model %s : Mean Score : %0.5f" % (k, v))

print()

print('Conv Scores : ', sum(conv_scores) / len(conv_scores))
print('N-Conv Scores : ', sum(n_conv_scores) / len(n_conv_scores))
print('LSTM Scores : ', sum(lstm_scores) / len(lstm_scores))



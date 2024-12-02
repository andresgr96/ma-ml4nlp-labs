from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from gensim.models import KeyedVectors
import numpy as np
import argparse
import os

from utils import *

def add_prev_tag_feature(data, targets=None, predicted_tags=None):
    """
    Adds the `prev_predicted_tag` feature to the feature dictionaries.
    """
    data_with_prev_tag = []
    for i, features in enumerate(data):
        new_features = features.copy()
        if i == 0:
            new_features['prev_predicted_tag'] = "<START>"
        else:
            if targets is not None:  # Training phase
                new_features['prev_predicted_tag'] = targets[i - 1]
            elif predicted_tags is not None:  # Inference phase
                new_features['prev_predicted_tag'] = predicted_tags[i - 1]
            else:
                raise ValueError("Either `targets` or `predicted_tags` must be provided.")
        data_with_prev_tag.append(new_features)
    return data_with_prev_tag

def prefit_vectorizer(train_data, feature_list):
    """
    Prefits a DictVectorizer with the training data.
    """
    vec = DictVectorizer()
    filtered_features = [
        {k: v for k, v in feature.items() if k in feature_list} 
        for feature in train_data
    ]
    vec.fit(filtered_features)
    return vec

def create_classifier(train_features, train_targets, vec, model_type='NB', max_iter=1000, use_embeddings=False, minibatch_size=10000):
    """
    Train a machine learning model (logreg, NB, SVM) for NER tags using minibatch learning.
    """
    # Vectorize categorical features
    categorical_features = vec.transform(train_features)
    if use_embeddings:
        embeddings = np.array([feature['embedding'] for feature in train_features])
        features_combined = np.hstack([categorical_features.toarray(), embeddings])
    else:
        features_combined = categorical_features

    if model_type == 'logreg':
        model = LogisticRegression(max_iter=max_iter)
        model.fit(features_combined, train_targets)
    elif model_type == 'NB':
        model = MultinomialNB()
        classes = np.unique(train_targets)
        features_combined, train_targets = shuffle(features_combined, train_targets, random_state=42)
        for start in range(0, len(train_targets), minibatch_size):
            end = start + minibatch_size
            batch_features = features_combined[start:end]
            batch_targets = train_targets[start:end]
            model.partial_fit(batch_features, batch_targets, classes=classes)
    elif model_type == 'SVM':
        model = LinearSVC(max_iter=max_iter)
        model.fit(features_combined, train_targets)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model

def classify_data(model, vec, inputdata, outputfile, feature_list, embeddings, features_to_remove, word_embedding_model):
    """
    Classifies data using the prefitted vectorizer and writes results to a file.
    """
    features, _ = extract_features_and_labels(inputdata, word_embedding_model=word_embedding_model)
    predictions = []
    prev_predicted_tag = "<START>"

    if embeddings:
        features = remove_specified_features(features=features, features_to_remove=features_to_remove)
        embeddings_feats = np.array([feature['embedding'] for feature in features])
    else:
        embeddings_feats = None

    features_no_embeddings = remove_embedding_from_features(features)
    for i in range(len(features_no_embeddings)):
        features_no_embeddings[i]['prev_predicted_tag'] = prev_predicted_tag
        filtered_features = [
            {k: v for k, v in feature.items() if k in feature_list} 
            for feature in features_no_embeddings
        ]
        categorical_features = vec.transform(filtered_features)

        if embeddings:
            features_combined = np.hstack([categorical_features.toarray(), embeddings_feats])
        else:
            features_combined = categorical_features

        # Predict one token at a time
        prediction = model.predict(features_combined[i:i + 1])[0]
        predictions.append(prediction)
        prev_predicted_tag = prediction

    with open(outputfile, 'w') as outfile:
        counter = 0
        for line in open(inputdata, 'r'):
            components = line.rstrip('\n').split()
            if components:
                token, pos_tag, chunk_tag = components[:3]
                outfile.write(f"{token} {pos_tag} {chunk_tag} {predictions[counter]}\n")
                counter += 1

def main(args):
    os.makedirs(args.results_path, exist_ok=True)
    pred_file_base = os.path.join(args.results_path, "predictions")

    print("Loading word embeddings...")
    word_embedding_model = KeyedVectors.load_word2vec_format(args.embeddings_file, binary=True)

    print("Extracting features and labels...")
    train_data, train_targets = extract_features_and_labels(args.train_file, word_embedding_model)
    dev_data, dev_targets = extract_features_and_labels(args.dev_file, word_embedding_model)
    test_data, test_targets = extract_features_and_labels(args.test_file, word_embedding_model)

    print("Adding `prev_predicted_tag` to training data...")
    train_data = add_prev_tag_feature(train_data, targets=train_targets)
    dev_data = add_prev_tag_feature(dev_data, targets=dev_targets)

    feature_list = [
        'token', 'pos_tag', 'chunk_tag', 'capitalized', 'contains_digit',
        'word_frequency', 'token_length_bin', 'prev_pos_tag', 'next_pos_tag',
        'prefix_2', 'suffix_2', 'prefix_3', 'suffix_3', 'prev_predicted_tag', 'embedding'
    ]

    features_to_remove = ['token', 'capitalized', 'token_length', 'word_frequency', 'token_length_bin',
                          'prefix_2', 'suffix_2', 'prefix_3', 'suffix_3']

    print("Prefitting vectorizer...")
    train_data_no_embeddings = remove_embedding_from_features(train_data)
    vec = prefit_vectorizer(train_data_no_embeddings, feature_list[:-1])

    print("Training and evaluating models...")
    for model_name in ['logreg', 'NB', 'SVM']:
        print(f"Training: {model_name}")
        if model_name == "SVM":
            train_data_svm = remove_specified_features(train_data, features_to_remove)
            model = create_classifier(train_features=train_data_svm, train_targets=train_targets,
                                      vec=vec, model_type=model_name, max_iter=10000, use_embeddings=True)
        else:
            
            model = create_classifier(train_features=train_data_no_embeddings, train_targets=train_targets,
                                      vec=vec, model_type=model_name, max_iter=10000, use_embeddings=False)

        pred_file = f"{pred_file_base}_{model_name}.conll"
        print(f"Evaluating model: {model_name}")
        classify_data(
            model=model, vec=vec, inputdata=args.test_file, outputfile=pred_file,
            feature_list=feature_list, embeddings=(model_name == "SVM"),
            features_to_remove=features_to_remove, word_embedding_model=word_embedding_model
        )
        print(f"Evaluation results for {model_name}:")
        if args.evaluation == "token":
            evaluate_ner(args.test_file, pred_file)
        else:
            span_based_evaluation(args.test_file, pred_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NER Training and Evaluation")
    parser.add_argument("train_file", type=str, nargs="?", default="./data/conll2003/conll2003.train.conll", help="Path to the training file")
    parser.add_argument("dev_file", type=str, nargs="?", default="./data/conll2003/conll2003.dev.conll", help="Path to the development file")
    parser.add_argument("test_file", type=str, nargs="?", default="./data/conll2003/conll2003.test.conll", help="Path to the test file")
    parser.add_argument("results_path", type=str, nargs="?", default="./code/assignment2/results/", help="Directory to save results")
    parser.add_argument("embeddings_file", type=str, nargs="?", default="./data/vecs/GoogleNews-vectors-negative300.bin.gz", help="Path to the Word2Vec embeddings file")
    parser.add_argument("--evaluation", choices=["token", "span"], default="token", help="Evaluation type: token or span")
    parser.add_argument("--mode", choices=["default", "hyperparam"], default="default", help="Mode of operation: train 3 models or hyperparameter tuning")
    args = parser.parse_args()

    main(args)

from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import random
import sys
import os


def extract_features_and_labels(trainingfile):
    """
    Helper function to parse a document in CONLL format and build a feature dictionary for each token.
    This version extracts additional features for hypothesis testing and training.
    """ 
    data = []
    targets = []
    with open(trainingfile, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split() # just extract more features in this function to avoid more loops down the road
            if len(components) > 0:  
                token = components[0]
                pos_tag = components[1]
                chunk_tag = components[2]
                ner_tag = components[3]

                feature_dict = {
                    'token': token,
                    'pos_tag': pos_tag,
                    'chunk_tag': chunk_tag,
                    'capitalized': token[0].isupper(),
                    'contains_digit': any(char.isdigit() for char in token),
                    'token_length': len(token)
                }
                
                data.append(feature_dict)
                targets.append(ner_tag)
    return data, targets

def create_dataframe(data, targets):
    """Helper function to turn the datasets into dataframes"""   
    records = []
    for features, target in zip(data, targets):
        record = features.copy()
        record['NER'] = target
        records.append(record)
    
    return pd.DataFrame(records)

def read_labels(file_path):
    """
    Helper function to part a CONLL file and extract only the labels,
    """ 
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            components = line.strip().split()
            if components:  
                labels.append(components[-1])  
    return labels


def evaluate_ner(gt_file, pred_file):
    """
    Function to compare the true and predicted labels of two CONLL formatted files and calculate model metrics.
    """ 
    gt_labels = read_labels(gt_file)
    pred_labels = read_labels(pred_file)

    if len(gt_labels) != len(pred_labels):
        raise ValueError("Ground truth and prediction files must have the same number of labeled tokens.")

    labels = sorted(set(gt_labels + pred_labels)) 
    cm = confusion_matrix(gt_labels, pred_labels, labels=labels)

    precision = precision_score(gt_labels, pred_labels, labels=labels, average='weighted', zero_division=0)
    recall = recall_score(gt_labels, pred_labels, labels=labels, average='weighted', zero_division=0)
    f1 = f1_score(gt_labels, pred_labels, labels=labels, average='weighted', zero_division=0)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

def extract_spans(labels):
    """
    Function to extract spans from BIO-labeled sequences of tokens.
    Each span is represented as (start_index, end_index, label).
    """
    spans = []
    start = None
    current_label = None

    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current_label is not None:
                spans.append((start, i - 1, current_label))
            start = i
            current_label = label[2:]
        elif label.startswith("I-") and current_label == label[2:]:
            continue
        else:
            if current_label is not None:
                spans.append((start, i - 1, current_label))
            current_label = None
            start = None

    if current_label is not None:
        spans.append((start, len(labels) - 1, current_label))

    return spans

def span_based_evaluation(gt_file, pred_file, check_spans=False):
    """
    Function to evaluate precision, recall, F1-score, and plot a confusion matrix at the span level.

    Setting 'check_spans' to True will print the first 5 spans, can be used to compare with the file for verification
    """
    
    gt_labels = read_labels(gt_file)
    pred_labels = read_labels(pred_file)
    if len(gt_labels) != len(pred_labels):
        raise ValueError("Ground truth and prediction files must have the same number of tokens.")

    gt_spans = extract_spans(gt_labels)
    pred_spans = extract_spans(pred_labels)
    label_counts = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    
    for span in gt_spans:
        if span in pred_spans:
            label_counts[span[2]]["TP"] += 1  
        else:
            label_counts[span[2]]["FN"] += 1  

    for span in pred_spans:
        if span not in gt_spans:
            label_counts[span[2]]["FP"] += 1  

    all_labels = sorted(set(label for _, _, label in gt_spans + pred_spans))
    precisions, recalls, f1_scores = {}, {}, {}

    for label in all_labels:
        tp = label_counts[label]["TP"]
        fp = label_counts[label]["FP"]
        fn = label_counts[label]["FN"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precisions[label] = precision
        recalls[label] = recall
        f1_scores[label] = f1

        print(f"{label} - Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

    matrix = []
    for true_label in all_labels:
        row = []
        for pred_label in all_labels:
            if true_label == pred_label:
                row.append(label_counts[true_label]["TP"])
            else:
                row.append(label_counts[pred_label]["FP"] if pred_label in label_counts else 0)
        matrix.append(row)

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=all_labels, yticklabels=all_labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Span-Level Confusion Matrix")
    plt.show()

    overall_tp = sum(counts["TP"] for counts in label_counts.values())
    overall_fp = sum(counts["FP"] for counts in label_counts.values())
    overall_fn = sum(counts["FN"] for counts in label_counts.values())

    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    overall_f1 = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    print(f"\nOverall - Precision: {overall_precision:.2f}, Recall: {overall_recall:.2f}, F1-Score: {overall_f1:.2f}")

    if check_spans:
        print(f"GT Spans: {list(gt_spans)[:5]}")


def create_classifier(train_features, train_targets, max_iter):
    """
    Function to train a logistic regression model to predict NER tags.
    
    Uses a vectorizer to transform key-value like features into vectors.
    """
    logreg = LogisticRegression(max_iter=max_iter)
    vec = DictVectorizer()
    features_vectorized = vec.fit_transform(train_features)
    model = logreg.fit(features_vectorized, train_targets)
    
    return model, vec


def classify_data(model, vec, inputdata, outputfile, feature_list):
    """Classifies data using only specified features and writes results to a file."""
    features, _ = extract_features_and_labels(inputdata)
    filtered_features = [{k: v for k, v in feature.items() if k in feature_list} for feature in features]
    print(f"Features used: {filtered_features[0]}") # Sanity check
    features_vectorized = vec.transform(filtered_features)
    predictions = model.predict(features_vectorized)
    
    # Using "with" will automatically close the file after it snot used anymore btw
    with open(outputfile, 'w') as outfile:
        counter = 0
        for line in open(inputdata, 'r'):
            components = line.rstrip('\n').split()
            if components:
                # Write token and predicted label, excluding the original true label, that way we can still use our evaluation functions that require two different files
                token, pos_tag, chunk_tag = components[:3]
                outfile.write(f"{token} {pos_tag} {chunk_tag} {predictions[counter]}\n")
                counter += 1


def main(args):
    train_data, train_targets = extract_features_and_labels(args.train_file)
    test_data, test_targets = extract_features_and_labels(args.test_file)
    dev_data, dev_targets = extract_features_and_labels(args.dev_file)

    feature_list = ['token', 'pos_tag', 'chunk_tag', 'capitalized', 'contains_digit']   # Define features for model
    ml_model, vec = create_classifier(train_data, train_targets, max_iter=10000)
    
    classify_data(model=ml_model, vec=vec, inputdata=args.dev_file, outputfile=args.pred_file, feature_list=feature_list)

    # Evaluate based on the chosen method
    if args.token_eval:
        evaluate_ner(args.dev_file, args.pred_file)
    else:
        span_based_evaluation(args.dev_file, args.pred_file, check_spans=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER Model Training and Evaluation")

    parser.add_argument("--train_file", type=str, default="../../data/conll2003/conll2003.train.conll",
                        help="Path to the training file (default: '../../data/conll2003/conll2003.train.conll')")
    parser.add_argument("--test_file", type=str, default="../../data/conll2003/conll2003.test.conll",
                        help="Path to the test file (default: '../../data/conll2003/conll2003.test.conll')")
    parser.add_argument("--dev_file", type=str, default="../../data/conll2003/conll2003.dev.conll",
                        help="Path to the dev file (default: '../../data/conll2003/conll2003.dev.conll')")
    parser.add_argument("--pred_file", type=str, default="./results/log_results.conll",
                        help="Path to save the prediction results (default: './results/log_results.conll')")
    parser.add_argument("--token_eval", action="store_true",
                        help="If set, perform token-level evaluation; otherwise, use span-level evaluation")

    args = parser.parse_args()
    
    main(args)


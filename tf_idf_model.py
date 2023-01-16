from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizer
import transformers
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import pandas as pd
import string
from nltk.corpus import stopwords
from transformers import logging
import math

logging.set_verbosity_error()


def get_percent_change(cur, prev):
    if cur == prev:
        return 0
    try:
        return (abs(cur - prev) / prev) * 100.0
    except ZeroDivisionError:
        return 0


def indicators(argument_labels: pd.DataFrame, tf_idf_values: pd.DataFrame, threshold):
    indicator_tokens = {}
    for index, label in enumerate(list(argument_labels.columns[1:])):
        indicator_tokens[label] = []
        values = tf_idf_values[label].sort_values(ascending=False)  # len 7077
        for jndex, v in enumerate(values):
            if v > threshold:
                indicator_tokens[label].append(values.index[jndex])
            else:
                break
    return indicator_tokens


def predict_individual(premise: str, tf_idf_values: pd.DataFrame, argument_labels: pd.DataFrame, threshold: float = 0.00005):
    class_indicators = indicators(argument_labels, tf_idf_values, threshold)
    # print(class_indicators)
    prediction = [0 for _ in argument_labels.columns[1:]]
    premise = premise.translate(str.maketrans('', '', string.punctuation))
    # for token in premise.split():
    #     token = token.lower()
    for index, label in enumerate(list(argument_labels.columns[1:])):
        if any(i in premise.split() for i in class_indicators[label]):
            prediction[index] = 1
        # print(label, token, class_indicators[label])
        # if token in class_indicators[label]:
        #     prediction[index] = 1
    # print(premise.split())
    return prediction


def evaluate_individual(argument_id: str, prediction: list, argument_labels: pd.DataFrame):
    # print(prediction)
    # print(list(argument_labels.loc[argument_labels["Argument ID"] == argument_id].iloc[0, :])[1:])
    # print(manhattan(prediction, list(argument_labels.loc[argument_labels["Argument ID"] == argument_id].iloc[0, :])[1:]))
    return manhattan(prediction, list(argument_labels.loc[argument_labels["Argument ID"] == argument_id].iloc[0, :])[1:])


def manhattan(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a, b))


def score_model(true_labels: pd.DataFrame, predicted_labels: list, label: str):
    # scores = []
    # for index, label in enumerate(list(true_labels.columns[1:])):
    y_values = list(true_labels[label].values)
    y_hat = []
    index = list(true_labels.columns[1:]).index(label)
    for i in predicted_labels:
        y_hat.append(i[index])
    print(label)
    print(y_values)
    print(y_hat)
    # print(precision_recall_fscore_support(y_values, y_hat))
    scores = precision_recall_fscore_support(y_values, y_hat, average='weighted')
    # scores.append(precision_recall_fscore_support(y_values, y_hat))
    return scores


if __name__ == '__main__':
    """
    for label in list(labels.columns[1:]):
        print(label)
        curr = arguments[labels[label] == 1]  # filter by current label
        count = 0
        for arg_id, text in zip(list(curr["Argument ID"]), list(curr["Premise"])):
            for token in text.split():
                if token in list(tf_idf_scores.index) and tf_idf_scores.loc[token, label] > 0:
                    print(tf_idf_scores.loc[token, label])

    percent_tracking = [False for _ in range(len(labels.columns[1:]))]
    counts = [0 for _ in range(len(labels.columns[1:]))]
    """
    # print(list(labels.loc[<ARG ID HERE>, :])[1:])  # get true labels of argument
    """
    for index, label in enumerate(list(labels.columns[1:])):
        values = tf_idf_scores[label].sort_values(ascending=False)  # len 7077
        # print(list(values.index)[0])
        # print(values)
        # for i in range(len(values) - 1):
        #     delta = get_percent_change(values[i + 1], values[i])
        #     if delta > 5 and i < 200:
        #         # print(f"{label} % change: {delta}  |  Index {i+1}")
        #         percent_tracking[index] = True
        #         counts[index] += 1
        for jndex, v in enumerate(values):
            if v > 0.00005:
                counts[index] = counts[index] + 1  # randomly selected 0.0005 based on tf-idf csv file
            else:
                print(jndex)
                break

    # print(sum(percent_tracking))
    print(counts)
    """

    arguments = pd.read_csv("SemEval/data/arguments-training.tsv", sep='\t')
    tf_idf_scores = pd.read_csv("SemEval/tf_idf_final.csv", index_col='Unnamed: 0')
    labels = pd.read_csv("SemEval/data/labels-training.tsv", sep='\t')

    with open("tf_idf_results.txt", 'w') as tf_idf_file:
        for thresh in [0.00006, 0.00005, 0.00004, 0.00003]:
            tf_idf_file.write(f"Threshold: {format(thresh, '.8f')}")
            results = []
            incorrect = []
            predictions = []
            for arg_id, text in zip(list(arguments["Argument ID"]), list(arguments["Premise"])):
                text_pred = predict_individual(text.lower(), tf_idf_scores, labels, thresh)
                predictions.append(text_pred)
                evaluation = evaluate_individual(arg_id, text_pred, labels)
                results.append(evaluation)
                if evaluation != 0:
                    incorrect.append(evaluation)
                # break

            tf_idf_file.write(f"\tNum records: {len(results)}")
            tf_idf_file.write(f"\tNum classified incorrectly: {len(incorrect)}")
            tf_idf_file.write(f"\t{(((len(incorrect)) / len(results)) * 100).__round__(4)}% incorrect\n")
        tf_idf_file.write("--- Precision, Recall, and F1 Score ---\n")
        for label in list(labels.columns[1:]):
            p_r_f1 = score_model(labels, predictions, label)
            print(f"{label}\n\tPrecision: {p_r_f1[0]}\n\tRecall: {p_r_f1[1]}\n\tF1-Score: {p_r_f1[2]}\n")
            tf_idf_file.write(f"{label}\n\tPrecision: {p_r_f1[0]}\n\tRecall: {p_r_f1[1]}\n\tF1-Score: {p_r_f1[2]}\n")

import json
import os
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformers
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine
from sklearn.metrics import precision_recall_fscore_support
from transformers import BertConfig, BertModel, BertTokenizer, logging

logging.set_verbosity_error()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
stop_words = set(stopwords.words("english"))


def find_and_save_vectors(arguments, labels, outfile: str = 'vectors'):
    for label in list(labels.columns[1:]):
        all_vectors = {}
        print(label.lower().split()[-1])
        curr = arguments[labels[label] == 1]  # filter by label
        for arg_id, text in zip(list(curr["Argument ID"]), list(curr["Premise"])):
            marked_text = "[CLS] " + text + " [SEP]"
            tokenized_text = tokenizer.tokenize(marked_text)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = [1] * len(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])
            model = BertModel.from_pretrained(
                "bert-base-uncased",
                output_hidden_states=True,  # Whether the model returns all hidden-states.
            )
            model.eval()
            with torch.no_grad():
                outputs = model(tokens_tensor, segments_tensors)
                hidden_states = outputs[2]
            token_embeddings = torch.stack(hidden_states, dim=0)
            token_embeddings = torch.squeeze(token_embeddings, dim=1)
            token_embeddings = token_embeddings.permute(1, 0, 2)

            token_vecs_sum = []
            # For each token in the sentence...
            for token in token_embeddings:
                sum_vec = torch.sum(token[-4:], dim=0)
                token_vecs_sum.append(sum_vec)

            for i, token_str in enumerate(tokenized_text):
                # print(f"{token_str}_{arg_id}_{i}")  # word, arg, and position
                all_vectors[f"{token_str}_{arg_id}_{i}"] = token_vecs_sum[i].tolist()
        with open(f"SemEval/JSON/bert_{outfile}_{label.lower().split()[-1]}.json", "w") as outfile:
            json.dump(all_vectors, outfile, indent=4)


def get_percent_change(cur, prev):
    if cur == prev:
        return 0
    try:
        return (abs(cur - prev) / prev) * 100.0
    except ZeroDivisionError:
        return 0


def predict_individual(
    premise: str,
    condensed_cosine: dict,
    argument_labels: pd.DataFrame,
    threshold: float = 0.9,
):
    prediction = [0 for _ in argument_labels.columns[1:]]
    for index, label in enumerate(list(argument_labels.columns[1:])):
        curr = condensed_cosine[label.lower().split()[-1]].loc[
            condensed_cosine[label.lower().split()[-1]]["Cosine"]
            >= condensed_cosine[label.lower().split()[-1]].Cosine.quantile(threshold)
        ]
        if any(i in premise.split() for i in list(curr.index)):
            prediction[index] = 1
    return prediction


def evaluate_individual(
    argument_id: str, prediction: list, argument_labels: pd.DataFrame
):
    # print(prediction)
    # print(list(argument_labels.loc[argument_labels["Argument ID"] == argument_id].iloc[0, :])[1:])
    # print(manhattan(prediction, list(argument_labels.loc[argument_labels["Argument ID"] == argument_id].iloc[0, :])[1:]))
    return manhattan(
        prediction,
        list(
            argument_labels.loc[argument_labels["Argument ID"] == argument_id].iloc[
                0, :
            ]
        )[1:],
    )


def manhattan(a, b):
    return sum(abs(val1 - val2) for val1, val2 in zip(a, b))


def condense(to_condense: pd.DataFrame):
    new_df = pd.DataFrame(columns=to_condense.columns)
    for index in to_condense.index:
        # print(new_df)
        if index.split("_")[0] not in new_df.index:
            new_df.loc[index.split("_")[0]] = to_condense.loc[index, "Cosine"]
    return new_df


def score_model(true_labels: pd.DataFrame, predicted_labels: list, label: str):
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
        scores = precision_recall_fscore_support(y_values, y_hat, average="weighted")
        # scores.append(precision_recall_fscore_support(y_values, y_hat))
    return scores


def retrieve_embedding(text: str, argument_id: str):
    """
    :param text: A sentence to be passed though bert in order to retrieve an embedding
    :rtype text: string
    :param argument_id: The unique argument id of the text parameter
    :rtype argument_id: string
    :return: bert embedding for the input sentence
    """
    # Add the special tokens.
    marked_text = "[CLS] " + text + " [SEP]"

    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)

    # convert input to torch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    model = BertModel.from_pretrained('bert-base-uncased',
                                      output_hidden_states=True,  # Whether the model returns all hidden-states.
                                      )

    model.eval()

    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers.
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)

        # Evaluating the model will return a different number of objects based on
        # how it's  configured in the `from_pretrained` call earlier. In this case,
        # becase we set `output_hidden_states = True`, the third item will be the
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1, 0, 2)
    # Stores the token vectors, with shape [22 x 768]
    token_vecs_sum = []

    # `token_embeddings` is a [22 x 12 x 768] tensor.

    # For each token in the sentence...
    for token in token_embeddings:
        # `token` is a [12 x 768] tensor
        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-4:], dim=0)

        # Use `sum_vec` to represent `token`.
        token_vecs_sum.append(sum_vec)

    text_dict = {}
    for i, token_str in enumerate(tokenized_text):
        # print(f"{token_str}_{arg_id}_{i}")  # word, arg, and position
        text_dict[f"{token_str}_{argument_id}_{i}"] = token_vecs_sum[i].tolist()
        # print(i, token_str)  # position, word
        # print(token_vecs_sum[i])  # vector
    # print(text_dict)
    return text_dict


def cosine_model():
    pass


if __name__ == "__main__":
    # find_and_save_vectors()
    """
    for label in list(labels.columns[1:]):
        print(label)
        curr = arguments[labels[label] == 1]  # filter by current label
        count = 0
        for arg_id, text in zip(list(curr["Argument ID"]), list(curr["Premise"])):
            for token in text.split():
                if token in list(cosine_scores.index) and cosine_scores.loc[token, label] > 0:
                    print(cosine_scores.loc[token, label])
        
    percent_tracking = [False for _ in range(len(labels.columns[1:]))]
    counts = [0 for _ in range(len(labels.columns[1:]))]
    # print(list(labels.loc[<ARG ID HERE>, :])[1:])  # get true labels of argument
    for index, label in enumerate(list(labels.columns[1:])):
    values = cosine_scores[label].sort_values(ascending=False)  # len 7077
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
    """
    # load all cosine calculations and store results in dictionary
    condensed_labels = {}
    for label in list(labels.columns[1:]):
        condensed_labels[label.lower().split()[-1]] = condense(
            pd.read_csv(
                f"SemEval/Cosine/complete_{label.lower().split()[-1]}_cosine.csv",
                index_col="Token",
            )
        )
    # break
    with open("cosine_results.txt", "w") as cosine_file:
        cosine_file.write("--- Evaluating classes based on percentile ---\n")
        for percentile in [0.99, 0.95, 0.9, 0.8]:
            cosine_file.write(f"Percentile: {percentile * 100}%")
            results = []
            incorrect = []
            predictions = []
            for arg_id, text in zip(
                list(arguments["Argument ID"]), list(arguments["Premise"])
            ):
                text = text.translate(str.maketrans("", "", string.punctuation)).lower()
                # print(arg_id, text)
                text_pred = predict_individual(
                    text.lower(), condensed_labels, labels, percentile
                )
                predictions.append(text_pred)
                evaluation = evaluate_individual(arg_id, text_pred, labels)
                results.append(evaluation)
                if evaluation != 0:
                    incorrect.append(evaluation)
            # break
        # break

    cosine_file.write(f"\tNum records: {len(results)}")
    cosine_file.write(f"\tNum classified incorrectly: {len(incorrect)}")
    cosine_file.write(
        f"\t{(((len(incorrect)) / len(results)) * 100).__round__(4)}% incorrect\n"
    )
    cosine_file.write("--- Precision, Recall, and F1 Score ---\n")
    for label in list(labels.columns[1:]):
        p_r_f1 = score_model(labels, predictions, label)
        print(
            f"{label}\n\tPrecision: {p_r_f1[0]}\n\tRecall: {p_r_f1[1]}\n\tF1-Score: {p_r_f1[2]}\n"
        )
        cosine_file.write(
            f"{label}\n\tPrecision: {p_r_f1[0]}\n\tRecall: {p_r_f1[1]}\n\tF1-Score: {p_r_f1[2]}\n"
        )
    """
    # evaluation script separate from prediction
    # use args in cosine files?
    # direct text mapping
    #    for every token in test premise
    #        embed and compare to top x words
    # save embeddings from dict with json.dump
    # train-dev split (80-20)

    # get embeddings of train & test
    # compare train embeddings to dev embeddings and get cosine, if above certain threshold assign label else 0
    training_set = pd.read_csv("SemEval/data/arguments-training.tsv", sep="\t")
    training_labels = pd.read_csv("SemEval/data/labels-training.tsv", sep="\t")
    dev_set = pd.read_csv("SemEval/data/arguments-validation.tsv", sep="\t")
    dev_labels = pd.read_csv("SemEval/data/labels-validation.tsv", sep="\t")
    # retrieve training embeddings
    training_embeddings = {}
    for each_label in list(training_labels.columns[1:]):
        print(each_label)
        with open(f"SemEval/JSON/bert_vectors_{each_label.lower().split()[-1]}.json") as json_file:  # load all embeddings from json files
            training_embeddings[each_label.lower().split()[-1]] = json.load(json_file)
        break
    for arg_id, text in zip(list(dev_set["Argument ID"]), list(dev_set["Premise"])):
        embeddings = retrieve_embedding(text, arg_id)  # bert embedding for ``text``
        prediction = [0 for _ in dev_labels.columns[1:]]
        for index, label in enumerate(list(dev_labels.columns[1:])):
            cosine_values = pd.read_csv(f"SemEval/Cosine/complete_{label.lower().split()[-1]}_cosine.csv", index_col="Token")  # load cosine file
            cosine_bound = min(cosine_values.loc[cosine_values["Cosine"] >= cosine_values.Cosine.quantile(0.99)].values)[0]  # top 1% of cosine scores of label
            # print(cosine_bound)
            # compare text embeddings to all embeddings in training set for specific label
            print(embeddings.values())
            
            for dev_embed in list(embeddings.values()):
                for value in list(training_embeddings.values()):
                    print(len(dev_embed))
                    print(len(value))
                    print(len(value[0]))
                    print(cosine(dev_embed, value))
                    if cosine(dev_embed, value) >= cosine_bound:
                        prediction[index] = 1
                        break
            # print(dev_labels.columns)
            # curr = dev_labels[label]
            # print(curr)
            # curr = curr.loc[curr["Cosine"] >= curr.Cosine.quantile(0.99)]
            # print(curr)
            # if any(i in premise.split() for i in list(curr.index)):
            #     prediction[index] = 1
        # print(prediction)
        # break

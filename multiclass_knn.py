import json
import os

import pandas as pd
import torch
from nltk.corpus import stopwords
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from transformers import BertModel, BertTokenizer, logging

import MultiKNN

logging.set_verbosity_error()
# sentence bert -> or cls from regular bert
# compare sentence embeddings of test to sentence embeddings of train
# count occurrences of each class
# if above threshold -> test premise gets that label


def store_labels(y_true: pd.DataFrame, dataset_descriptor: str) -> dict:
    """
    Takes a Dataframe and converts it to a dictionary, and stores it in a JSON file, or retrieves it from a JSON file if already saved
    :param y_true: The dataframe containing the labels
    :param dataset_descriptor: Describes the data we are storing/loading
    :return: A dictionary of the format { "argument ID" : [ true label for argument ] }
    """
    if os.path.exists(f"SemEval/JSON/true_labels_{dataset_descriptor}.json"):
        with open(
            f"SemEval/JSON/true_labels_{dataset_descriptor}.json", "r"
        ) as filename:
            label_dict = json.load(filename)
    else:
        label_dict = {}
        for r in range(len(y_true.index)):
            vector = list(y_true.loc[r])
            label_dict[vector[0]] = [
                int(vector[i]) for i in range(len(vector)) if i > 0
            ]
        with open(
            f"SemEval/JSON/true_labels_{dataset_descriptor}.json", "w"
        ) as filename:
            json.dump(label_dict, filename)
    return label_dict


def save_embeddings(df, dataset_descriptor):
    to_save = {}
    if (
        dataset_descriptor.lower() != "validation"
        and dataset_descriptor.lower() != "training"
    ):
        raise ValueError(
            f"{dataset_descriptor} must be either 'validation' or 'training' --- Try Again."
        )
    elif os.path.exists(f"SemEval/JSON/{dataset_descriptor.lower()}_CLS.json"):
        print(f"{dataset_descriptor.capitalize()} already completed")
    elif dataset_descriptor.lower() == "validation":
        for arg_id, text in zip(list(df["Argument ID"]), list(df["Premise"])):
            # print(arg_id, text)
            # Add the special tokens.
            marked_text = "[CLS] " + text + " [SEP]"

            # Split the sentence into tokens.
            tokenized_text = tokenizer.tokenize(marked_text)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = [1] * len(tokenized_text)

            # convert input to torch tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])

            model = BertModel.from_pretrained(
                "bert-base-uncased",
                output_hidden_states=True,  # Whether the model returns all hidden-states.
            )

            model.eval()

            # Run the text through BERT, and collect all hidden states produced from all 12 layers.
            with torch.no_grad():
                outputs = model(tokens_tensor, segments_tensors)
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

            for i, token_str in enumerate(tokenized_text):
                if token_str == "[CLS]":
                    # print([float(j) for j in token_vecs_sum[i]])
                    to_save[arg_id] = [float(j) for j in token_vecs_sum[i]]
        with open(
            f"SemEval/JSON/{dataset_descriptor.lower()}_CLS.json", "w"
        ) as filename:
            json.dump(to_save, filename)
        print(f"Save {dataset_descriptor.lower()} embeddings to JSON file")
    elif dataset_descriptor.lower() == "training":
        for arg_id, text in zip(list(df["Argument ID"]), list(df["Premise"])):
            # print(arg_id, text)
            # Add the special tokens.
            marked_text = "[CLS] " + text + " [SEP]"

            # Split the sentence into tokens.
            tokenized_text = tokenizer.tokenize(marked_text)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = [1] * len(tokenized_text)

            # convert input to torch tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])

            model = BertModel.from_pretrained(
                "bert-base-uncased",
                output_hidden_states=True,  # Whether the model returns all hidden-states.
            )

            model.eval()

            # Run the text through BERT, and collect all hidden states produced from all 12 layers.
            with torch.no_grad():
                outputs = model(tokens_tensor, segments_tensors)
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

            for i, token_str in enumerate(tokenized_text):
                if token_str == "[CLS]":
                    # print([float(j) for j in token_vecs_sum[i]])
                    to_save[arg_id] = [float(j) for j in token_vecs_sum[i]]
        with open(
            f"SemEval/JSON/{dataset_descriptor.lower()}_CLS.json", "w"
        ) as filename:
            json.dump(to_save, filename)
        print(f"Save {dataset_descriptor.lower()} embeddings to JSON file")
    # elif dataset_descriptor.lower() == 'training':
    #     json_files = [f for f in os.listdir("SemEval/JSON") if f.endswith('.json')]
    #     for filename in json_files:
    #         with open(os.path.join("SemEval/JSON", filename)) as f:
    #             data = json.load(f)
    #             for key in data.keys():
    #                 if key.split("_")[0] == '[CLS]':
    #                     to_save[key.split("_")[1]] = data[key]
    #     print(f"Save {dataset_descriptor.lower()} embeddings to JSON file")
    #     with open(f'SemEval/JSON/{dataset_descriptor.lower()}_CLS.json', 'w') as filename:
    #         json.dump(to_save, filename)


def generate_dataset(dataset_descriptor):
    curr_dataset = []
    # load json for descriptor CLS token
    with open(f"SemEval/JSON/{dataset_descriptor.lower()}_CLS.json", "r") as filename:
        arguments = json.load(filename)
    # load labels
    labels = store_labels(
        pd.read_csv(f"SemEval/labels-{dataset_descriptor}.tsv", sep="\t"),
        dataset_descriptor,
    )
    # store in list of tuples -> [(embedding, labels), (...), (...), ... ]
    for arg_id in arguments.keys():
        curr_dataset.append((arguments[arg_id], labels[arg_id]))
    # return the newly built dataset
    return curr_dataset


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
stop_words = set(stopwords.words("english"))

training_arguments = pd.read_csv("SemEval/data/arguments-training.tsv", sep="\t")
validation_arguments = pd.read_csv("SemEval/data/arguments-validation.tsv", sep="\t")
training_labels = pd.read_csv("SemEval/data/labels-training.tsv", sep="\t")
validation_labels = pd.read_csv("SemEval/data/labels-validation.tsv", sep="\t")

print("Validation in progress")
# save_embeddings(validation_arguments, 'validation')
print("Validation complete\nTraining in progress")
# save_embeddings(training_arguments, 'training')
print("Training complete")

print("\tDataset for Validation")
validation_dataset = generate_dataset("validation")
print("\tDataset for Training")
training_dataset = generate_dataset("training")

THRESHOLD: float = 0.4  # Percentage (as float) of how many records must be of a given label to be labeled as that clas = 5

# record structure -> [ embedding: list, labels: list ]
dataset = [
    [[2.7810836, 2.550537003], [0, 1, 0]],
    [[1.465489372, 2.362125076], [0, 1, 1]],
    [[3.396561688, 4.400293529], [0, 1, 1]],
    [[1.38807019, 1.850220317], [0, 1, 0]],
    [[3.06407232, 3.005305973], [0, 1, 1]],
    [[7.627531214, 2.759262235], [1, 1, 0]],
    [[5.332441248, 2.088626775], [1, 1, 0]],
    [[6.922596716, 1.77106367], [1, 1, 0]],
    [[8.675418651, -0.242068655], [1, 1, 0]],
    [[7.673756466, 3.508563011], [1, 0, 1]],
]

thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
accuracy_frame = pd.DataFrame(columns=thresholds)
precision_frame = pd.DataFrame(columns=thresholds)
recall_frame = pd.DataFrame(columns=thresholds)
f1_frame = pd.DataFrame(columns=thresholds)
test_data = validation_dataset[:1]
for neighbours in range(5, 36, 5):
    print(neighbours)
    for thresh in thresholds:
        knn = MultiKNN.Multi_Label_KNN(
            training_data=training_dataset, k=neighbours, threshold=thresh
        )
        predictions = knn.predict(test_dataset=test_data)  # predict validation
        print(predictions[0])
        # print("Predictions")
        # print(len(predictions))
        # print(len(validation_dataset[:5]))
        actual = [arg[1] for arg in test_data]  # true validation
        # print("Actual")
        # print(actual)
        # evaluated = MultiKNN.evaluate(predictions, actual)  # compare true and predicted
        # print("Evaluated")
        # print(evaluated)
        acc = accuracy_score(y_true=actual, y_pred=predictions)  # get accuracy
        precision = precision_score(y_true=actual, y_pred=predictions, average="micro")
        recall = recall_score(y_true=actual, y_pred=predictions, average="micro")
        f1 = f1_score(y_true=actual, y_pred=predictions, average="micro")
        # print(f"Accuracy for k={neighbours} and threshold={thresh} -> {acc}")
        # print(f"Precision for k={neighbours} and threshold={thresh} -> {precision}")
        # print(f"Recall for k={neighbours} and threshold={thresh} -> {recall}")
        # print(f"F1 Score for k={neighbours} and threshold={thresh} -> {f1}")
        accuracy_frame.loc[f"{neighbours}", thresh] = acc
        precision_frame.loc[f"{neighbours}", thresh] = precision
        recall_frame.loc[f"{neighbours}", thresh] = recall
        f1_frame.loc[f"{neighbours}", thresh] = f1
accuracy_frame.to_csv("multi_knn_accuracy_35.csv")
precision_frame.to_csv("multi_knn_precision_35.csv")
recall_frame.to_csv("multi_knn_recall_35.csv")
f1_frame.to_csv("multi_knn_f1_35.csv")

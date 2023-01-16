import json
import os

import pandas as pd
import torch
from nltk.corpus import stopwords
from transformers import BertModel, BertTokenizer, logging

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
stop_words = set(stopwords.words("english"))
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


def save_embeddings(dataset_descriptor):
    print(dataset_descriptor)
    df = pd.read_csv(f"SemEval/arguments-{dataset_descriptor}.tsv", sep="\t")
    print(df.head(5))
    to_save = {}
    if (
        dataset_descriptor.lower() != "validation"
        and dataset_descriptor.lower() != "training"
        and dataset_descriptor.lower() != "test"
    ):
        raise ValueError(
            f"{dataset_descriptor} must be either 'validation', 'training', or 'test --- Try Again."
        )
    elif os.path.exists(f"SemEval/JSON/{dataset_descriptor.lower()}_tokens.json"):
        print(f"{dataset_descriptor.capitalize()} already completed")
    elif dataset_descriptor.lower() == "validation":
        for arg_id, text in zip(list(df["Argument ID"]), list(df["Premise"])):
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
                    to_save[f"{token_str}_{arg_id}_{i}"] = [float(j) for j in token_vecs_sum[i]]
        with open(
            f"SemEval/JSON/{dataset_descriptor.lower()}_tokens.json", "w"
        ) as filename:
            json.dump(to_save, filename)
        print(f"Save {dataset_descriptor.lower()} embeddings to JSON file")
    elif dataset_descriptor.lower() == "training":
        for arg_id, text in zip(list(df["Argument ID"]), list(df["Premise"])):
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
                to_save[f"{token_str}_{arg_id}_{i}"] = [float(j) for j in token_vecs_sum[i]]
        with open(
            f"SemEval/JSON/{dataset_descriptor.lower()}_tokens.json", "w"
        ) as filename:
            json.dump(to_save, filename)
        print(f"Save {dataset_descriptor.lower()} embeddings to JSON file")
    elif dataset_descriptor.lower() == "test":
        for arg_id, text in zip(list(df["Argument ID"]), list(df["Premise"])):
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
                if token_str == '[CLS]':
                    to_save[f"{token_str}_{arg_id}_{i}"] = [float(j) for j in token_vecs_sum[i]]
        with open(
                f"SemEval/JSON/{dataset_descriptor.lower()}_tokens.json", "w"
        ) as filename:
            json.dump(to_save, filename)
        print(f"Save {dataset_descriptor.lower()} embeddings to JSON file")


if __name__ == "__main__":
    # save_embeddings("training")
    # save_embeddings("validation")
    save_embeddings("test")

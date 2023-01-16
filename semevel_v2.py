import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizer
import transformers
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import pandas as pd
import string
from nltk.corpus import stopwords

# print(transformers.__version__)

# https://huggingface.co/docs/transformers/model_doc/bert
# https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#31-running-bert-on-our-text

# configuration = BertConfig()
# model = BertModel(configuration)
# configuration = model.config
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

stop_words = set(stopwords.words('english'))
# print(stop_words)

arguments = pd.read_csv("SemEval/data/arguments-training.tsv", sep='\t')
labels = pd.read_csv("SemEval/data/labels-training.tsv", sep='\t')

thought = arguments[labels["Self-direction: thought"] == 1]  # filter by thought

thought_strings = []
thought_vectors = []

count = 0
for arg_id, text in zip(list(thought["Argument ID"]), list(thought["Premise"])):
    print(arg_id, text)
    # Add the special tokens.
    marked_text = "[CLS] " + text + " [SEP]"

    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)

    # convert input to torch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # print(tokens_tensor)
    # print(segments_tensors)

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

    for i, token_str in enumerate(tokenized_text):
        # print(f"{token_str}_{arg_id}_{i}")  # word arg and position
        thought_strings.append(f"{token_str}_{arg_id}_{i}")
        thought_vectors.append(token_vecs_sum[i])
        # print(i, token_str)  # position, word
        # print(token_vecs_sum[i])  # vector
    # # print("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
    # # layer_i = 0
    # #
    # # print("Number of batches:", len(hidden_states[layer_i]))
    # # batch_i = 0
    # #
    # # print("Number of tokens:", len(hidden_states[layer_i][batch_i]))
    # # token_i = 0
    # #
    # # print("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))
    #
    # # For the 5th token in our sentence, select its feature values from layer 5.
    # # token_i = 5
    # # layer_i = 5
    # # vec = hidden_states[layer_i][batch_i][token_i]
    #
    # # Plot the values as a histogram to show their distribution.
    # plt.figure(figsize=(10, 10))
    # plt.hist(vec, bins=200)
    # plt.show()
    if count > 2:
        break
    else:
        count += 1

center_of_venn_diagram = pd.DataFrame(columns=['Word 1', 'Word 2', 'Cosine'])
bookkeeping = dict()
for i in range(len(thought_strings)):  # get each embedding in current argument
    print(thought_strings[i])
    if thought_strings[i].split("_")[0] not in stop_words and thought_strings[i].split("_")[0] != "[SEP]" and thought_strings[i].split("_")[0] != "[CLS]" and thought_strings[i].split("_")[0] not in string.punctuation:
        if f"{thought_strings[i].split('_')[0]}_token" not in bookkeeping.keys():
            bookkeeping[f"{thought_strings[i].split('_')[0]}_{thought_strings[i].split('_')[1]}_{thought_strings[i].split('_')[2]}_token"] = 0
            bookkeeping[f"{thought_strings[i].split('_')[0]}_{thought_strings[i].split('_')[1]}_{thought_strings[i].split('_')[2]}_cls"] = 0
        token_cosine = []
        cls_cosine = []

        for j in range(len(thought_strings)):  # get all other embeddings
            if i == j or thought_strings[i].split("_")[1] == thought_strings[j].split("_")[1]:  # ensure we don't check vectors from same premise
                pass
            elif thought_strings[i].split("_")[0] == "[SEP]" or thought_strings[i].split("_")[0] == "[CLS]" or thought_strings[i].split("_")[0] in string.punctuation:
                pass
            elif thought_strings[j].split("_")[0] == "[SEP]" or thought_strings[j].split("_")[0] in string.punctuation or thought_strings[j].split("_")[0].split("'")[0] in stop_words:
                pass
            elif thought_strings[j].split("_")[0] == "[CLS]":
                # print(f"Compare CLS: {thought_strings[i], thought_strings[j], cosine(thought_vectors[i], thought_vectors[j])}")
                cls_cosine.append(1 - cosine(thought_vectors[i], thought_vectors[j]))
            else:
                # print(f"Compare Token: {thought_strings[i], thought_strings[j], cosine(thought_vectors[i], thought_vectors[j])}")
                token_cosine.append(1 - cosine(thought_vectors[i], thought_vectors[j]))
                """
                if cosine(thought_vectors[i], thought_vectors[j]) > 0.90:
                    # if thought_strings[j] not in visited:
                    # print(thought_strings[i], thought_strings[j], cosine(thought_vectors[i], thought_vectors[j]))
                    print(f"Comparer: {thought_strings[i].split('_')[0]} | Comparee: {thought_strings[j].split('_')[0]} | Cosine: {cosine(thought_vectors[i], thought_vectors[j])}")
                    # visited.append(thought_strings[i])
                    # center_of_venn_diagram.concat([thought_strings[i], thought_strings[j], float(cosine(thought_vectors[i], thought_vectors[j]))])
                    # center_of_venn_diagram.append([np.array([thought_strings[i], thought_strings[j]]), float(cosine(thought_vectors[i], thought_vectors[j]))])
                    center_of_venn_diagram = center_of_venn_diagram.append({"Word 1": thought_strings[i], "Word 2": thought_strings[j], "Cosine": float(cosine(thought_vectors[i], thought_vectors[j]))}, ignore_index=True)
                """
        if len(token_cosine) > 0 and len(cls_cosine) > 0:
            bookkeeping[f"{thought_strings[i].split('_')[0]}_{thought_strings[i].split('_')[1]}_{thought_strings[i].split('_')[2]}_token"] = (bookkeeping[f"{thought_strings[i].split('_')[0]}_{thought_strings[i].split('_')[1]}_{thought_strings[i].split('_')[2]}_token"] + (sum(token_cosine)/len(token_cosine))) / 2
            bookkeeping[f"{thought_strings[i].split('_')[0]}_{thought_strings[i].split('_')[1]}_{thought_strings[i].split('_')[2]}_cls"] = (bookkeeping[f"{thought_strings[i].split('_')[0]}_{thought_strings[i].split('_')[1]}_{thought_strings[i].split('_')[2]}_cls"] + (sum(cls_cosine)/len(cls_cosine))) / 2
            # center_of_venn_diagram = center_of_venn_diagram.append({"Word 1": thought_strings[i].split("_")[0], "Word 2": "Token", "Cosine": sum(token_cosine)/len(token_cosine)},
            #                                                        ignore_index=True)
            # center_of_venn_diagram = center_of_venn_diagram.append({"Word 1": thought_strings[i].split("_")[0], "Word 2": "CLS", "Cosine": sum(cls_cosine)/len(cls_cosine)},
            #                                                        ignore_index=True)
    # print(bookkeeping)

# parallelization

# change threshold
# tf-idf (prayer in thought vs other classes)
# token embedding to every other arg cls embedding -> avg over -> store
###
# xgboost
# multiclass knn
# naive bayes

# final = center_of_venn_diagram.sort_values("Cosine", ascending=False).reset_index().drop(columns=['index'])
# final = pd.DataFrame(bookkeeping.items(), columns=['Token', 'Cosine'])
# final = final[final["Cosine"] != 1].reset_index().drop(columns=['index'])
final = pd.DataFrame(bookkeeping.items(), columns=['Token', 'Cosine']).sort_values("Cosine", ascending=False).reset_index().drop(columns=['index'])
# final.to_csv("SemEval/Cosine/thought_token_cls_cosine_test_3.csv", index=False)
print(final.head(10))
# final = pd.DataFrame(bookkeeping.items(), columns=['Token', 'Cosine'])

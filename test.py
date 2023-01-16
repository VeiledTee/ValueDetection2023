import torch
from transformers import BertConfig, BertModel, BertTokenizer
import transformers
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import pandas as pd
import string

# print(transformers.__version__)

# https://huggingface.co/docs/transformers/model_doc/bert
# https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#31-running-bert-on-our-text

# configuration = BertConfig()
# model = BertModel(configuration)
# configuration = model.config
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

arguments = pd.read_csv("data/arguments.tsv", sep='\t')
labels = pd.read_csv("data/labels-level2.tsv", sep='\t')

thought = arguments[labels["Self-direction: thought"] == 1]  # filter by thought

print(len(thought.index))
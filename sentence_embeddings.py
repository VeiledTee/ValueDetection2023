import pandas as pd
from sentence_transformers import SentenceTransformer

# get model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# get data
training_labels = pd.read_csv("SemEval/data/labels-training.tsv", sep="\t").drop(columns=['Argument ID'])
validation_labels = pd.read_csv("SemEval/data/labels-validation.tsv", sep="\t").drop(columns=['Argument ID'])
training_data = pd.read_csv("SemEval/data/arguments-training.tsv", sep="\t")
validation_data = pd.read_csv("SemEval/data/arguments-validation.tsv", sep="\t")

# format data and labels
train_labels = []
for index, row in training_labels.items():
    train_labels.append(list(row))

valid_labels = []
for index, row in validation_labels.items():
    valid_labels.append(list(row))

train_premise = [text for text in training_data['Premise']]
validation_premise = [text for text in validation_data['Premise']]

# embed premises
train_embeddings = model.encode(train_premise)
validation_embeddings = model.encode(validation_premise)
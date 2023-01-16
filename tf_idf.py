import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizer
import transformers
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


def term_frequency(doc: str, word: str):
    doc = doc.split()
    return np.log10((doc.count(word) / len(doc)) + 1)


def in_doc(doc: str, word: str):
    return 1 if word in doc.split() else 0


def inverse_document_frequency(word: str):
    N = 0  # num docs "word" appears in
    for document in documents:
        N += in_doc(document[0], word)
    return np.log10(len(documents) / N)  # log_10(total num documents / num docs containing word)


def tf_idf(document, term):
    print(f"Term: {term}")
    print(f"TF: {term_frequency(document, term)}")
    print(f"IDF: {inverse_document_frequency(term)}")
    print(f"Count: {document.split().count(term)}")

    return term_frequency(document, term) * inverse_document_frequency(term)


stop_words = set(stopwords.words('english'))

# build documents to perform tf-idf on
documents = []
arguments = pd.read_csv("data/arguments.tsv", sep='\t')
labels = pd.read_csv("data/labels-level2.tsv", sep='\t')

cols = list(labels.columns)[1:]
tf_idf_results = pd.DataFrame(columns=cols)

for i in range(1, len(labels.columns)):
    curr = []
    curr_arg = arguments[labels[labels.columns[i]] == 1]  # filter by current argument
    doc_text = ''
    for text in list(curr_arg["Premise"]):
        curr.append(text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        doc_text += text.lower() + " "
    print(len(curr))
    documents.append([doc_text, set(doc_text.split())])

for i in range(len(documents)):
    for t in documents[i][1]:
        skip = False
        if t in stop_words:
            pass
        else:
            if t not in list(tf_idf_results.index):  # create empty row full of 0 values for unseen (and not "stop") words
                tf_idf_results.loc[t, list(tf_idf_results.columns)] = list(np.zeros(len(cols)))
            tf_idf_results.loc[t, cols[i]] = tf_idf(documents[i][0], t)

tf_idf_results.to_csv("tf_idf_final", index=True)

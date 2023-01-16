import xgboost
import json
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from save_results import export_results
from sentence_transformers import SentenceTransformer
"""
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# have to do each class individually
training_labels = pd.read_csv("data/labels-training.tsv", sep="\t").drop(columns=['Argument ID'])
validation_labels = pd.read_csv("data/labels-validation.tsv", sep="\t").drop(columns=['Argument ID'])

training_data = pd.read_csv("data/arguments-training.tsv", sep="\t")
validation_data = pd.read_csv("data/arguments-validation.tsv", sep="\t")

train_labels = []
for index, row in training_labels.items():
    train_labels.append(list(row))

valid_labels = []
for index, row in validation_labels.items():
    valid_labels.append(list(row))

train_premise = [text for text in training_data['Premise']]
validation_premise = [text for text in validation_data['Premise']]

train_embeddings = model.encode(train_premise)
validation_embeddings = model.encode(validation_premise)
"""

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# have to do each class individually
training_labels = pd.concat([pd.read_csv("data/labels-training.tsv", sep="\t").drop(columns=['Argument ID']), pd.read_csv("data/labels-validation.tsv", sep="\t").drop(columns=['Argument ID'])]).reset_index(drop=True)
training_data = pd.concat([pd.read_csv("data/arguments-training.tsv", sep="\t"), pd.read_csv("data/arguments-validation.tsv", sep="\t")]).reset_index(drop=True)

test_data = pd.read_csv("data/arguments-test.tsv", sep="\t")

train_labels = []
for index, row in training_labels.items():
    train_labels.append(list(row))

train_premise = [text for text in training_data['Premise']]
# validation_premise = [text for text in validation_data['Premise']]
test_premise = [text for text in test_data['Premise']]

train_embeddings = model.encode(train_premise)
test_embeddings = model.encode(test_premise)


def xgboost_sentence_tuning(parameters):
    # Split the data into features and labels
    output = []
    # f1_values = []
    # precision_values = []
    # recall_values = []
    for i in range(len(training_labels.columns)):
        print(training_labels.columns[i])
        # Convert the data into DMatrix format
        dmatrix = xgboost.DMatrix(data=train_embeddings, label=train_labels[i])

        # Train the XGBoost model
        bst = xgboost.train(parameters, dmatrix)

        # Make predictions
        prediction = list(bst.predict(xgboost.DMatrix(test_embeddings)))
        round_pred = [round(pred) for pred in prediction]

        # Save prediction to dataframe
        output.append(round_pred)

        # Evaluate
        # f1 = f1_score(list(valid_labels[i]), round_pred)
        # precision = precision_score(list(valid_labels[i]), round_pred)
        # recall = recall_score(list(valid_labels[i]), round_pred)
        #
        # f1_values.append(f1)
        # precision_values.append(precision)
        # recall_values.append(recall)

    # print(f"Avg F1-Score: {np.mean(f1_values)}")
    # print(f"Avg Precision: {np.mean(precision_values)}")
    # print(f"Avg Recall: {np.mean(recall_values)}")
    #
    # with open('output/MD Files/xgboost-sentence.md', 'a') as f:
    #     f.write(
    #         f"\n| ```'sentence-transformers/all-MiniLM-L6-v2'``` | ```'{xgboost_params['objective']}'``` | {xgboost_params['eta']} | {xgboost_params['max_depth']} | {xgboost_params['subsample']} | {xgboost_params['colsample_bytree']} | {xgboost_params['lambda']} | {np.mean(f1_values)} | {np.mean(precision_values)} | {np.mean(recall_values)} |\n"
    #     )

    export_results(list(training_labels.columns.insert(0, "Argument ID")), list(test_data['Argument ID']), output, 'output/xgboost/xgboost_sentence_final.tsv')


if __name__ == '__main__':
    param_list = [{
        'objective': 'binary:logistic',
        'eta': 0.5,
        'max_depth': 15,
        'subsample': 0.15,
        'colsample_bytree': 0.5,
        'lambda': 0.25,
    }]
    for xgboost_params in param_list:
        xgboost_sentence_tuning(xgboost_params)

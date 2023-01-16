import xgboost
import json
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from save_results import export_results

TF_IDF_THRESHOLD = 10

tf_idf = pd.read_csv('tf_idf_final.csv', index_col='Unnamed: 0')
# print(list(tf_idf.index[:10]))

print('Load train')
with open(f"JSON/training_tokens.json") as filename:
    category_data = [v for k, v in json.load(filename).items() if k.split("_")[0] == '[CLS]']
print('Load validation')
with open(f"JSON/validation_tokens.json") as filename:
    test_data = []
    test_arguments = []
    for k, v in json.load(filename).items():
        # category_data.append(v)
        test_data.append(v)
        test_arguments.append(k)
"""
print('Load Test')
with open(f"JSON/test_tokens.json") as filename:
    test_data = []
    test_arguments = []
    for k, v in json.load(filename).items():
        test_data.append(v)
        test_arguments.append(k.split("_")[1])
"""

training_args = pd.concat([pd.read_csv("data/arguments-training.tsv", sep="\t"), pd.read_csv("data/arguments-validation.tsv", sep="\t")]).reset_index(drop=True)
train_frame = pd.read_csv("data/arguments-training.tsv", sep="\t")
valid_frame = pd.read_csv("data/arguments-validation.tsv", sep="\t")
test_frame = pd.read_csv("data/arguments-test.tsv", sep="\t")

train_premise = [text for text in training_args['Premise']]
test_premise = [text for text in valid_frame['Premise']]
# test_premise = [text for text in test_frame['Premise']]


def xgboost_tuning(parameters):
    # training_labels = pd.concat([pd.read_csv("data/labels-training.tsv", sep="\t"),
    #                              pd.read_csv("data/labels-validation.tsv", sep="\t")]).reset_index(drop=True)
    training_labels = pd.read_csv("data/labels-training.tsv", sep="\t")
    validation_labels = pd.read_csv("data/labels-validation.tsv", sep="\t")

    # results_frame = pd.DataFrame(columns=training_labels.columns)
    # results_frame[training_labels.columns[0]] = test_arguments

    # output = []
    f1_values = []
    precision_values = []
    recall_values = []
    # Split the data into features and labels
    for category in training_labels.columns[1:]:
        print(category)
        sorted_tf_idf = tf_idf.sort_values(by=[category])
        to_compare = list(sorted_tf_idf.index[:TF_IDF_THRESHOLD])
        # Convert the data into DMatrix format
        dmatrix = xgboost.DMatrix(data=category_data, label=training_labels[category])

        # Train the XGBoost model
        bst = xgboost.train(parameters, dmatrix)

        # Make predictions
        prediction = bst.predict(xgboost.DMatrix(test_data))
        round_pred = [round(pred) for pred in prediction]

        for i in range(len(test_premise)):
            if any(x in test_premise[i] for x in to_compare):
                round_pred[i] = 1

        # Save prediction to dataframe
        # output.append(round_pred)

        # Evaluate
        f1_values.append(f1_score(list(validation_labels[category].values), round_pred))
        precision_values.append(precision_score(list(validation_labels[category].values), round_pred))
        recall_values.append(recall_score(list(validation_labels[category].values), round_pred))

    print(f"Avg F1-Score: {np.mean(f1_values)}")
    print(f"Avg Precision: {np.mean(precision_values)}")
    print(f"Avg Recall: {np.mean(recall_values)}")

    with open('output/MD Files/xgboost-uncased.md', 'a') as f:
        f.write(
            f"| ```'bert-base-uncased'``` | {TF_IDF_THRESHOLD} ({np.round((TF_IDF_THRESHOLD / 7077) * 100, decimals=4)}%) | {xgboost_params['eta']} | {xgboost_params['max_depth']} | {xgboost_params['subsample']} | {xgboost_params['colsample_bytree']} | {xgboost_params['lambda']} | {np.mean(f1_values)} | {np.mean(precision_values)}  | {np.mean(recall_values)}  |\n"
        )

    # export_results(training_labels.columns, test_arguments, output, 'output/xgboost/xgboost_uncased_final.tsv')


if __name__ == '__main__':
    param_list = [{
        'objective': 'binary:logistic',
        'eta': 0.5,
        'max_depth': 15,
        'subsample': 0.25,
        'colsample_bytree': 1,
        'lambda': 0,
    }, {
        'objective': 'binary:logistic',
        'eta': 0.5,
        'max_depth': 15,
        'subsample': 0.5,
        'colsample_bytree': 1,
        'lambda': 1,
    }, {
        'objective': 'binary:logistic',
        'eta': 0.5,
        'max_depth': 15,
        'subsample': 0.25,
        'colsample_bytree': 0,
        'lambda': 0,
    }, {
        'objective': 'binary:logistic',
        'eta': 0.5,
        'max_depth': 25,
        'subsample': 0.25,
        'colsample_bytree': 1,
        'lambda': 0,
    }, {
        'objective': 'binary:logistic',
        'eta': 0.5,
        'max_depth': 15,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'lambda': 0.5,
    }]
    for xgboost_params in param_list:
        xgboost_tuning(xgboost_params)

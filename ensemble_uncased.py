import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from save_results import export_results

print('Load train')
with open(f"JSON/training_tokens.json") as filename:
    category_data = [v for k, v in json.load(filename).items() if k.split("_")[0] == '[CLS]']
print('Load validation')
with open(f"JSON/validation_tokens.json") as filename:
    # validation_data = []
    validation_arguments = []
    for k, v in json.load(filename).items():
        category_data.append(v)
        # validation_arguments.append(k)
print('Load Test')
with open(f"JSON/test_tokens.json") as filename:
    test_data = []
    test_arguments = []
    for k, v in json.load(filename).items():
        test_data.append(v)
        test_arguments.append(k.split("_")[1])


def ensemble_sans_case(ensemble_params):
    clf1 = LogisticRegression(multi_class=ensemble_params['multi_class'], max_iter=ensemble_params['max_iter'], n_jobs=-1)
    clf2 = RandomForestClassifier(n_estimators=ensemble_params['n_estimators'], max_depth=ensemble_params['max_depth'], n_jobs=-1)
    clf3 = GaussianNB()
    eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting=ensemble_params['voting'])

    # training_labels = pd.read_csv("data/labels-training.tsv", sep="\t")
    training_labels = pd.concat([pd.read_csv("data/labels-training.tsv", sep="\t"),
                                 pd.read_csv("data/labels-validation.tsv", sep="\t")]).reset_index(drop=True)

    results_frame = pd.DataFrame(columns=training_labels.columns)
    results_frame[training_labels.columns[0]] = test_arguments

    output = []
    # f1_values = []
    # precision_values = []
    # recall_values = []
    for category in training_labels.columns[1:]:
        print(category)
        category_labels = list(training_labels[category].values)

        eclf1 = eclf1.fit(category_data, category_labels)
        prediction = list(eclf1.predict(test_data))
        output.append(prediction)

        # Evaluate
        # f1 = f1_score(list(validation_labels[category].values), prediction)
        # precision = precision_score(list(validation_labels[category].values), prediction)
        # recall = recall_score(list(validation_labels[category].values), prediction)
        #
        # f1_values.append(f1)
        # precision_values.append(precision)
        # recall_values.append(recall)

    # with open('output/MD Files/ensemble-uncased.md', 'a') as f:
    #     f.write(
    #         f"| ```'bert-base-uncased'``` | ```'{ensemble_params['multi_class']}'``` | {ensemble_params['max_iter']} | {ensemble_params['max_depth']} | {ensemble_params['n_estimators']} | {ensemble_params['voting']} | {np.mean(f1_values)} | {np.mean(precision_values)}  | {np.mean(recall_values)}  |\n"
    #     )
    #
    # print(f"Avg F1-Score: {sum(f1_values) / len(f1_values)}")
    # print(f"Avg Precision: {sum(precision_values) / len(precision_values)}")
    # print(f"Avg Recall: {sum(recall_values) / len(recall_values)}")

    export_results(training_labels.columns, test_arguments, output, 'output/ensemble/ensemble_uncased_final.tsv')


if __name__ == '__main__':
    for params in [
        {
            'multi_class': 'multinomial',
            'max_iter': 100000,
            'max_depth': 500,
            'n_estimators': 500,
            'voting': 'soft',
        }
    ]:
        ensemble_sans_case(ensemble_params=params)

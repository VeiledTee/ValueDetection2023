# exactly how submitted
# output and etc
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier
import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from save_results import export_results
from sentence_transformers import SentenceTransformer


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# have to do each class individually
training_labels = pd.concat([pd.read_csv("data/labels-training.tsv", sep="\t").drop(columns=['Argument ID']), pd.read_csv("data/labels-validation.tsv", sep="\t").drop(columns=['Argument ID'])]).reset_index(drop=True)
training_data = pd.concat([pd.read_csv("data/arguments-training.tsv", sep="\t"),pd.read_csv("data/arguments-validation.tsv", sep="\t")]).reset_index(drop=True)

print('A01002' in training_data['Argument ID'].values)
print('A01001' in training_data['Argument ID'].values)

test_data = pd.read_csv("data/arguments-test.tsv", sep="\t")

train_labels = []
for index, row in training_labels.items():
    train_labels.append(list(row))

train_premise = [text for text in training_data['Premise']]
test_premise = [text for text in test_data['Premise']]
# validation_premise = [text for text in validation_data['Premise']]

train_embeddings = model.encode(train_premise)
test_embeddings = model.encode(test_premise)


def ensemble_tuning_sentence(ensemble_params):
    clf1 = LogisticRegression(multi_class=ensemble_params['multi_class'], max_iter=ensemble_params['max_iter'], n_jobs=-1)
    clf2 = RandomForestClassifier(n_estimators=ensemble_params['n_estimators'], max_depth=ensemble_params['max_depth'], n_jobs=-1)
    clf3 = GaussianNB()
    eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting=ensemble_params['voting'])

    output = []
    # f1_values = []
    # precision_values = []
    # recall_values = []
    for i in range(len(training_labels.columns)):
        print(training_labels.columns[i])
        eclf1 = eclf1.fit(train_embeddings, train_labels[i])
        prediction = list(eclf1.predict(test_embeddings))
        output.append(prediction)

        # Evaluate
        # f1 = f1_score(list(valid_labels[i]), prediction)
        # precision = precision_score(list(valid_labels[i]), prediction)
        # recall = recall_score(list(valid_labels[i]), prediction)

        # f1_values.append(f1)
        # precision_values.append(precision)
        # recall_values.append(recall)

    # with open('output/MD Files/ensemble-sentence.md', 'a') as f:
    #     f.write(
    #         f"| ```'sentence-transformers/all-MiniLM-L6-v2'``` | ```'{ensemble_params['multi_class']}'``` | {ensemble_params['max_iter']} | {ensemble_params['max_depth']} | {ensemble_params['n_estimators']} | {ensemble_params['voting']} | {np.mean(f1_values)} | {np.mean(precision_values)}  | {np.mean(recall_values)}  |\n"
    #     )

    # print(f"Avg F1-Score: {np.mean(f1_values)}")
    # print(f"Avg Precision: {np.mean(precision_values)}")
    # print(f"Avg Recall: {np.mean(recall_values)}")
    print(test_data.head())
    export_results(list(training_labels.columns.insert(0, "Argument ID")), test_data["Argument ID"], output, 'output/ensemble/ensemble_sentence_final.tsv')


if __name__ == '__main__':
    print('hi')
    for params in [{
            'multi_class': 'multinomial',
            'max_iter': 5000,
            'max_depth': 100,
            'n_estimators': 50,
            'voting': 'soft',
        }
    ]:
        ensemble_tuning_sentence(ensemble_params=params)

import math


def evaluate(y_hat: list, y: list) -> list:
    """
    :param y_hat: Predicted labels
    :param y: True Labels
    :return: List of integers, 1 if prediction == true at a given index, else 0
    """
    evaluations = []
    for i, j in zip(y_hat, y):  # loop through labels and compare
        if i == j:
            evaluations.append(1)
        else:
            evaluations.append(0)
    return evaluations


def accuracy(evaluations: list) -> float:
    """
    Calculates the accuracy of predictions as a %
    :param evaluations: A list of 0s and 1s
    :return: The % of the evaluations list that are 1s
    """
    return sum(evaluations) / len(evaluations)


def euclidean_distance(row1, row2):
    distance = 0.0
    for x, y in zip(row1, row2):
        distance += (y - x) ** 2
    return math.sqrt(distance)


class Multi_Label_KNN:
    def __init__(self, training_data, k: int = 5, threshold: float = 0.4):
        self.train = training_data
        self.k = k
        self.threshold = threshold

    def get_neighbours(self, test_row):
        distances = list()
        for train_row in self.train:
            dist = euclidean_distance(test_row[0], train_row[0])
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        return [distances[i][0] for i in range(self.k)]

    def predict(self, test_dataset):
        test_predictions = []
        for test in test_dataset:
            neighbours = self.get_neighbours(test)
            output_vector = [row[-1] for row in neighbours]
            individual_prediction = []
            for i in range(len(output_vector[0])):
                count = 0
                for vector in output_vector:
                    count = count + 1 if vector[i] == 1 else count
                if count / len(output_vector[0]) > self.threshold:
                    individual_prediction.append(1)
                else:
                    individual_prediction.append(0)
            test_predictions.append(individual_prediction)
        return test_predictions

"""
Author: Lucas Barusek
Basic Implementation of k-nearest-neighbors
"""

import csv, sys
from math import sqrt

def read_data(filename, delimiter=",", has_header=True):
    """Reads data file into a data list and header."""
    data = []
    header = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            header = next(reader, None)
        for line in reader:
            example = [float(x) for x in line]
            data.append(example)

        return header, data

def convert_data_to_pairs(data, header):
    """Turns a data list of lists into a list of (attribute, target) pairs."""
    pairs = []
    for example in data:
        x = []
        for i, element in enumerate(example):
            if header[i].startswith("target"):
                y = int(element)
            else:
                x.append(element)
        pair = (x, y)
        pairs.append(pair)
    return pairs

def accuracy(training, test, k):
    """Accuracy of k-nearest neighbors alg."""

    true_positives = 0
    total = len(test)

    for (x, y) in test:
        class_prediction = k_nearest_neighbors(training, x, k)
        if class_prediction == y:
            true_positives += 1

    return (true_positives / total)


################################################################################
### k-nearest neighbors


def euclideanDistance(x, y):
    """ Compute Euclidean distance for two sets of values """
    assert (len(x) == len(y))
    distance = 0
    for i in range(len(x)):
        distance += (abs(x[i] - y[i]) ** 2)

    return sqrt(distance)


def mostFrequentClass(training):
    """ Returns the most frequent class of the k-nearest neighbors. If there is a 
    tie, remove the furthest neighbor until there is no tie """
    endIndex = len(training)
    tie = True
    while tie:
        # Compute mode of the classes found in our nearest neighbors
        classes = [y for (_, y) in training[: endIndex]]
        maxValue = max(set(classes), key=classes.count)
        copyOfList = [c for c in classes if c != maxValue]
        if len(copyOfList) == 0: # only one class is represented
            tie = False
        # we have a tie, so remove the furthest neighbor from consideration
        elif maxValue ==  max(set(copyOfList), key=copyOfList.count):
            endIndex -= 1
        # There is no tie, so we found our prediction
        else:
            tie = False

    return maxValue       


def k_nearest_neighbors(training, query, k):
    """Runs k-nearest neighbors algorithm on one test example.
    - training: the labeled training data, which is a list of (x, y) tuples.
                Each x is a list of input floats for the example, and y is an
                integer.
    - query: an input vector x without the known class y.
    - k: the number of neighbors to consider.
    - returns: class prediction, which is an integer."""

    # sort the list by euclidean distance, and return the most frequent class 
    # within the k-nearest neighbors
    training.sort(key = lambda case: euclideanDistance(case[0], query))
    return mostFrequentClass(training[ : k])


def main():

    # k = 9

    header, data = read_data(sys.argv[1], ",")
    pairs = convert_data_to_pairs(data, header)

    test_header, test_data = read_data(sys.argv[2], ",")
    test_pairs = convert_data_to_pairs(test_data, test_header)

    # print(k_nearest_neighbors(pairs, [3.9, -0.4], 4))
    # print(k_nearest_neighbors(pairs, [3.9, -1.2], 5))

    for k in range(1, 20, 2):
        acc = accuracy(pairs, test_pairs, k)
        print("accuracy({}) = {}".format(k, acc))



if __name__ == "__main__":
    main()

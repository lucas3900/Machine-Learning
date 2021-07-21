"""
Author: Lucas Barusek

Implementation of a neural Network
    - implements forward and back propagation, testing and accuracy
    - dynamically sized network
    - works for binary classification, multi-classification, and incrementer

Usage:
    1) Edit Global Variables for number of Epochs or k-value for cross-validation
    2) python3 neuralNet.py DATASET.csv
    3) Follow input Statements
"""

import csv, sys, random, math
from time import sleep


EPOCHS = 1000
K_VALUE = 3


def read_data(filename, delimiter=",", has_header=True):
    """Reads datafile using given delimiter. Returns a header and a list of
    the rows of data."""
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
        y = []
        for i, element in enumerate(example):
            if header[i].startswith("target"):
                y.append(element)
            else:
                x.append(element)
        pair = (x, y)
        pairs.append(pair)
    return pairs


################################################################################
### Neural Network code goes here


class NeuralNetwork:
    """ Class Representation for Neural Network  """
    def __init__(self, layers):
        """ Neural Network Initializer given a layer structure """
        self._layers = layers 
        self._numNodes = sum(self._layers)
        self._accuracyOnTraining = 0

        # initialize activations to have a 1 at index 0 for dummy weight
        self._activations = [1] + [0] * self._numNodes
        self._deltas = [0] * self._numNodes

        # intialize weights to be a small random number
        self._weights = [[0]*(self._numNodes) for _ in range (self._numNodes)]
        for row in range(len(self._weights)):
            for col in range(len(self._weights[0])):
                self._weights[row][col] = random.uniform(-.5, .5)

    def __str__(self):
        """ Prints out values for deltas, activations, and weights for neural network """
        weightsString = ""
        for layer in self._weights:
            weightsString += f"{str(layer)}\n"
        return f"Neural Network Values\nLayers: {self._layers}\nActivation: {self._activations}\n\
Deltas: {self._deltas}\nWeights:\n{weightsString}"

    def backPropagationLearning(self, trainingSet, incrementer=False, stochastic=False):
        """ Implementation of back propagation learning given a training set. Uses dynamic
        learning  that decreases each epoch. If incrementer is true, tests for correctness 
        instead of accuracy each epoch. If stochastic is true, picks random test cases
        each epoch as opposed to going through each test once """

        for epoch in range(EPOCHS):
            alpha = 1000 / (1000 + epoch)
            for num in range(len(trainingSet)):
                if stochastic:
                    (x, y) = random.choice(trainingSet)
                else:
                    (x, y) = trainingSet[num]
                
                self.forwardPropagate(x)

                # calculate delta values for output layer
                for nodeIndex in range(sum(self._layers[:-1]), self._numNodes):
                    activation = self._activations[nodeIndex+1]
                    self._deltas[nodeIndex] = activation * (1 - activation) * (y[nodeIndex-sum(self._layers[:-1])] - activation)

                # calculate delta values for hidden-layer Nodes
                nodeIndex = self._numNodes - self._layers[-1] - 1
                for nodeIndex in range(self._numNodes - self._layers[-1] - 1, self._layers[0] - 1, -1):
                    self._deltas[nodeIndex] = self._activations[nodeIndex+1] * (1 - self._activations[nodeIndex+1]) * \
                                 sum([self._weights[nodeIndex+1][j] * self._deltas[j] for j in self.getNextLayer(nodeIndex)])

                # calculate dummy and edge weights for the network
                for row in range(self._numNodes):
                    for col in range(self._numNodes):
                        self._weights[row][col] += (alpha * self._activations[row] * self._deltas[col])

            print(f"Accuracy for Epoch {epoch+1}:", self.testCases(trainingSet, incrementer=incrementer))
        
        # record the final accuracy on training data
        self._accuracyOnTraining = self.testCases(trainingSet, incrementer=incrementer)

    def forwardPropagate(self, inputs):
        """ Implementation of forward propagation """

        # make input nodes activations equal to input values
        for i in range(self._layers[0]+1):
            self._activations[i] = inputs[i]
        
        # determine activation value for each non-input node
        for nodeIndex in range(self._layers[0], self._numNodes):
            inValue = sum([self._activations[i+1]*self._weights[i+1][nodeIndex] 
                           for i in self.getPreviousLayer(nodeIndex)]) + \
                           self._weights[0][nodeIndex] # add in dummy weight for node
            self._activations[nodeIndex+1] = self.activationFunc(inValue)
        
    def activationFunc(self, x, softplus=False):
        """ Activation function for forward propagation. Either logistic function
        or softplus function """
        if softplus:
            return math.log(1 + math.e**x)
        try:
            denom = (1 + math.e ** -x)
        except OverflowError:
            return 0.0
        return 1.0 / denom

    def getPreviousLayer(self, node):
        """ Given a node index, return the range of indices that correspond 
        to the nodes in the previous layer """
        assert (node not in range(self._layers[0]))
        previousRange = range(0, self._layers[0])
        currNode = self._layers[0]
        for layer in self._layers[1:]:
            if node in range(currNode, currNode + layer):
                return previousRange
            previousRange = range(currNode, currNode + layer)
            currNode += layer
        return previousRange

    def getNextLayer(self, node):
        """ Given a node index, return the range of indices that correspond 
        to the nodes in the next layer """
        assert (node not in range(sum(self._layers[:-1]), sum(self._layers)))
        if len(self._layers) == 2:
            return range(self._layers[0], sum(self._layers))
        nextLayer = range(sum(self._layers[0:1]), sum(self._layers[0:2]))
        currNode = 0
        for layer in range(len(self._layers[:-1])):
            if node in range(currNode, currNode+self._layers[layer]):
                return nextLayer
            currNode += self._layers[layer]
            nextLayer = range(sum(self._layers[0:layer+2]), sum(self._layers[0:layer+3]))

        return nextLayer

    def testAccuracy(self, case):
        """ Test accuracy for binary and multi classification. Iterate through 
        each output node and take the absolute value difference between it and
        the corresponding output vector index. Then return an average of all
        the differences.  """
        self.forwardPropagate(case[0])
        firstOutNode = sum(self._layers[:-1]) + 1
        return sum([1 - abs(case[1][i] - self._activations[firstOutNode+i]) 
                    for i in range(len(case[1]))]) / len(case[1])

    def testCorrect(self, case, incrementer=False):
        """ Given a single test case, determine whether the neural network predicts
        it correctly or incorrectly . Returns 1 for correct and 0 for incorrect """
        self.forwardPropagate(case[0]) # propagate inputs to yield activation values
        if self._layers[-1] == 1 or incrementer: # binary classification or incrementer
            # return 1 if and only if all of the activations nodes match the 
            # output vector at each index
            outputActivations = self._activations[sum(self._layers[:-1]) + 1:]
            for i in range(len(outputActivations)):
                # count a node as 1 if it is above the .5 threshold
                predicted = 1 if outputActivations[i] >= 0.5 else 0
                if predicted != case[1][i]:
                    return 0 # all nodes match so network predicted correctly
            return 1
        else: # multiclass classification
            # return 1 if the output node with the highest activation corresponds
            # the the index in the output vector that is 1
            outputActivations = self._activations[sum(self._layers[:-1]) + 1:]
            predictedIndex = outputActivations.index(max(outputActivations))
            return 1 if case[1][predictedIndex] == 1 else 0

    def testCases(self, cases, testCorrect=False, incrementer=False):
        """ Given a set of cases, either test for correctness or accuracy, and return 
        an average percentage (either average accuracy or average percent correct) """
        accuracy = []
        for case in cases:
            if testCorrect or incrementer:
                accuracy.append(self.testCorrect(case, incrementer))
            else:
                accuracy.append(self.testAccuracy(case))

        return sum(accuracy) / len(accuracy)

    def getAccuracyOnTraining(self):
        """ returns final accuracy on training data """
        return self._accuracyOnTraining


def crossValidation(testCases, nn, k=3):
    """ Implementation of cross validation. Breaks test cases into k subsets.
    First subset will be used for training, and the others will be used for testing """
    assert (k != 1)
    random.shuffle(testCases)
    subsetSize = len(testCases) // k
    nn.backPropagationLearning(testCases[: subsetSize])
    startIndex = subsetSize; endIndex = subsetSize * 2
    totalAccuracy = []
    for i in range(k - 1):
        if i == k - 2:
            totalAccuracy.append(nn.testCases(testCases[startIndex : ]))
        else:
            totalAccuracy.append(nn.testCases(testCases[startIndex: endIndex]))
        startIndex = endIndex
        endIndex += subsetSize

    return sum(totalAccuracy) / len(totalAccuracy)


def getMeanStd(lst):
    """ Return (mean, std) for a given list """
    mean = sum(lst) / len(lst)
    variance = sum([((x - mean) ** 2) for x in lst]) / len(lst)
    return  (mean, variance ** 0.5)


def standardizeData(cases):
    """ standardize a given training set """
    for attr in range(1, len(cases[0][0])):
        attributes = [cases[case][0][attr] for case in range(len(cases))]
        mean, std = getMeanStd(attributes)
        for case in range(len(cases)):
            try:
                cases[case][0][attr] = (cases[case][0][attr] - mean) / std
            except ZeroDivisionError:
                continue


def getHiddenLayers(inputs, outputs):
    """ gets input for neural network structure """
    print(f"Training for Model with {inputs} input variables and {outputs} classes")
    numLayers = int(input("How many hidden layers? "))
    layers = []
    for i in range(numLayers):
        numNodes = int(input(f"How many nodes in hidden layer {i+1}? "))
        layers.append(numNodes)  

    return layers 


def main():
    """ main driver """ 
    header, data = read_data(sys.argv[1], ",")

    pairs = convert_data_to_pairs(data, header)

    # Note: add 1.0 to the front of each x vector to account for the dummy input
    training = [([1.0] + x, y) for (x, y) in pairs]

    standardizeData(training)

    numInputs = len(training[0][0]) - 1; numOutputs = len(training[0][1])
    nn = NeuralNetwork([numInputs] + getHiddenLayers(numInputs, numOutputs) + [numOutputs])

    incrementerInput = input("Is this an incrementer problem (y/N)? ")
    incrementor = False if incrementerInput.lower() in ['', 'n'] else True

    if incrementor:
        nn.backPropagationLearning(training, True)
        print("\nPercent Correct for all Cases:", nn.testCases(training, True, True), "\n")
    else:
        print("\nAccuracy on Unseen Data:", crossValidation(training, nn, K_VALUE))
        print("Percent Correct for all Cases:",nn.testCases(training, True), "\n")

    sleep(2)

    print(nn)
        
if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

def split_data(csv, train_ratio, random_val, scale=True):
    # Splitting values into results and training data
    result = csv.loc[:,'Outcome']
    data = csv.drop('Outcome', 'columns')
    x_train, x_test, y_train, y_test = train_test_split(data, result, train_size=train_ratio, random_state=random_val)

    if scale:
        # Scaling values
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    return [x_train, x_test, y_train, y_test]

def neural_network(data, hidden_layer_size, activation, solver, max_iter, shuffle = True, random_state = None):
    mlp = MLPClassifier(hidden_layer_size, activation=activation, solver=solver, max_iter=max_iter, shuffle=shuffle, random_state=random_state)
    mlp.fit(data[0], data[2])
    prediction = mlp.predict(data[1])
    return [prediction, confusion_matrix(data[3], prediction), mlp.score(data[1], data[3])]

def testCase1(data):
    # To test different activation and solver combinations
    activations = ["identity", "logistic", "tanh", "relu"]
    solvers = ["lbfgs", "sgd", "adam"]

    for activation in activations:
        for solver in solvers:
            print(neural_network(data, (7,7), activation, solver, 10000, False, 8))

def main():
    # Reading csv file into dataframe
    csv = pd.read_csv("diabetes.csv")

    # Splitting and scaling data
    data = split_data(csv, 0.8, 8, True)
    testCase1(data)

if __name__ == "__main__":
    main()
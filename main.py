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

def activation_solver_test(csv):
    # To test different activation and solver combinations
    activations = ["identity", "logistic", "tanh", "relu"]
    solvers = ["lbfgs", "sgd", "adam"]

    # Test variables
    num_test = 20
    max_iteration = 50000
    hidden_layer = (20, 20)
    train_ratio = 0.8
    compiled_results = [[] for i in range(len(activations))]
    plt.figure()

    # Run tests
    for test in range(num_test):
        print("Test run {} beginning".format(test+1))
        data = split_data(csv, train_ratio, test, True)
        for activation in activations:
            for solver in solvers:
                compiled_results[activations.index(activation)].append(neural_network(data, hidden_layer, activation, solver, max_iteration, False, test))
            print("Activation {} completed".format(activation.upper()))
        print("Test run {} completed".format(test+1))

    # Plotting Figures
    for i in range(len(activations)):
        avg_score = []

        for j in range(len(solvers)):
            avg_score.append((sum([result[2] for result in compiled_results[i][j::3]])/num_test))
        
        for j in range(len(solvers)):
            plt.text(j, avg_score[j], round(avg_score[j], 4))

        plt.bar(solvers, avg_score)
        plt.title(activations[i].capitalize())
        plt.ylabel("Mean Score")
        plt.xlabel("Solvers")
        plt.savefig("figures/activation_solver_test/{} with different solvers".format(activations[i].capitalize()),bbox_inches="tight")
        plt.show()
        plt.figure()

    print("Activation-Solver test completed")

def hidden_layer_test(csv):
    activation = "identity"
    solver = "lbfgs"

    num_test = 20
    max_layers = 5
    max_neuron = 40
    neuron_iterate = 1
    max_iteration = 50000
    results = [[[] for j in range(max_neuron//neuron_iterate)] for i in range(max_layers)]

    for test in range(num_test):
        print("Test run {} beginning".format(test+1))
        data = split_data(csv, 0.8, test, True)
        for num_layer in range(1, max_layers+1):
            print("Layer {} beginning".format(num_layer))
            for num_neuron in range(neuron_iterate, max_neuron+neuron_iterate, neuron_iterate):
                hidden_layer = tuple([num_neuron for layer in range(num_layer)])
                results[num_layer-1][num_neuron//neuron_iterate-1].append(neural_network(data, hidden_layer, activation, solver, max_iteration, False, test)[2])
                
            print("Layer {} completed".format(num_layer))
        print("Test run {} completed".format(test+1))
    
    plt.figure()
    for num_layer in range(max_layers):
        x = [(sum(results[num_layer][num_neuron])/num_test) for num_neuron in range(max_neuron//neuron_iterate)]
        y = [num_neuron for num_neuron in range(neuron_iterate, max_neuron+neuron_iterate, neuron_iterate)]

        
        plt.plot(y, x, "bo--")
        plt.title("{} Hidden Layer".format(num_layer+1))
        plt.ylabel("Mean Score")
        plt.xlabel("Number of Neuron each Layer")
        plt.savefig("figures/hidden_layer_test/{} Hidden Layer".format(num_layer+1))
        plt.show()
        plt.figure()
        
def main():
    # Reading csv file into dataframe
    csv = pd.read_csv("diabetes.csv")

    # activation_solver_test(csv)
    hidden_layer_test(csv)

if __name__ == "__main__":
    main()
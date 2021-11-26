import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# Function to split data into test and training data
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

# Generates neural network and runs fitting and prediction. Returns predictions, confusion matrix, and score
def neural_network(data, hidden_layer_size, activation, solver, max_iter, shuffle = True, random_state = None):
    mlp = MLPClassifier(hidden_layer_size, activation=activation, solver=solver, max_iter=max_iter, shuffle=shuffle, random_state=random_state)
    mlp.fit(data[0], data[2])
    prediction = mlp.predict(data[1])
    return [prediction, confusion_matrix(data[3], prediction), mlp.score(data[1], data[3])]

# Test different number of hidden layer and neurons in each layer
def hidden_layer_test(csv):
    # Best activation function and solver combination found
    activation = "identity"
    solver = "lbfgs"
    
    # Test variables
    num_test = 20
    train_ratio = 0.8
    max_layers = 6
    max_neuron = 60
    neuron_iterate = 1
    max_iteration = 50000
    results = [[[] for j in range(max_neuron//neuron_iterate)] for i in range(max_layers)]

    # Run test
    for test in range(num_test):
        print("Test run {} beginning".format(test+1))
        data = split_data(csv, train_ratio, test, True)
        for num_layer in range(1, max_layers+1):
            for num_neuron in range(neuron_iterate, max_neuron+neuron_iterate, neuron_iterate):
                hidden_layer = tuple([num_neuron for layer in range(num_layer)])
                results[num_layer-1][num_neuron//neuron_iterate-1].append(neural_network(data, hidden_layer, activation, solver, max_iteration, False, test)[2])
            print("Layer {} completed".format(num_layer))
        print("Test run {} completed\n".format(test+1))

    # Plotting results    
    plt.figure()
    for num_layer in range(max_layers):
        # Getting x and y values
        ## Calculating mean score of all tests
        y = [(sum(results[num_layer][num_neuron])/num_test) for num_neuron in range(max_neuron//neuron_iterate)]
        ## Getting neuron numbers based on neuron iteration and max neuron
        x = [num_neuron for num_neuron in range(neuron_iterate, max_neuron+neuron_iterate, neuron_iterate)]

        # Plotting
        plt.plot(x, y, "r.--")
        for i,j in zip(x,y):
            if j != y[i//neuron_iterate-2]:
                plt.annotate(str(round(j,4)),xy=(i,j))
                
        plt.title("{} Hidden Layer".format(num_layer+1))
        plt.ylabel("Mean Score")
        plt.xlabel("Number of Neuron each Layer")
        plt.savefig("figures/hidden_layer_test/{} Hidden Layer".format(num_layer+1),bbox_inches="tight")
        plt.show()
        plt.figure()

def main():
    # Reading csv file into dataframe
    csv = pd.read_csv("diabetes.csv")
    
    # Run test function
    hidden_layer_test(csv)

if __name__ == "__main__":
    main()

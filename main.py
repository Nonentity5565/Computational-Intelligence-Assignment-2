import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

plt.ion()

""" 
def visualise(mlp, ax=None):
    n_neurons = [len(layer) for layer in mlp.coefs_]
    n_neurons.append(mlp.n_outputs_)

    y_range = [0, max(n_neurons)]
    x_range = [0, len(n_neurons)]
    loc_neurons = [
        [[l, (n + 1) * (y_range[1] / (layer + 1))] for n in range(layer)]
        for l, layer in enumerate(n_neurons)
    ]
    x_neurons = [x for layer in loc_neurons for x, y in layer]
    y_neurons = [y for layer in loc_neurons for x, y in layer]

    # identify the range of weights
    weight_range = [
        min([layer.min() for layer in mlp.coefs_]),
        max([layer.max() for layer in mlp.coefs_]),
    ]

    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    ax.cla()
    ax.scatter(x_neurons, y_neurons, s=100, zorder=5)
    for l, layer in enumerate(mlp.coefs_):
        for i, neuron in enumerate(layer):
            for j, w in enumerate(neuron):
                ax.plot([loc_neurons[l][i][0], loc_neurons[l + 1][j][0]],[loc_neurons[l][i][1], loc_neurons[l + 1][j][1]], "white", linewidth=((w - weight_range[0]) / (weight_range[1] - weight_range[0]) * 5 + 0.2) * 1.2,)
                ax.plot([loc_neurons[l][i][0], loc_neurons[l + 1][j][0]],[loc_neurons[l][i][1], loc_neurons[l + 1][j][1]], "grey", linewidth=(w - weight_range[0]) / (weight_range[1] - weight_range[0]) * 5 + 0.2,)

"""
data = pd.read_csv("diabetes.csv")s

# split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    #Balls
)

# preprocessing -- scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# print(pd.DataFrame(X_train, columns=data.feature_names).describe().transpose())

# training ann model
mlp = MLPClassifier(hidden_layer_sizes=(3), max_iter=1000)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for _ in range(10):
    for _ in range(100):
        mlp.partial_fit(X_train, y_train, np.unique(data.target))

    visualise(mlp, ax)
    plt.pause(0.1)

# for _ in range(100):
#   mlp.partial_fit(X_train,y_train,np.unique(data.target))

# predictions = mlp.predict(X_test)
# print(confusion_matrix(y_test, predictions))

# for _ in range(1000):
#   mlp.partial_fit(X_train,y_train,np.unique(data.target))
#   print(mlp.n_iter_)

# predictions = mlp.predict(X_test)
# print(confusion_matrix(y_test, predictions))

mlp.fit(X_train, y_train)
print(mlp.n_iter_)

# prediction
predictions = mlp.predict(X_test)

# evaluation
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

visualise(mlp)

input()

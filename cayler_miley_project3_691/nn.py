import numpy as np
from matplotlib import pyplot as plt

'''
Calculates loss over all training samples on the given model. 

Params: 
    model: Neural network model with the same input dimensions
        as the number of features and the same output dimension
        as the number of classes described by y
    X: Array of data samples with arbitrary number of features
    y: Ground truth labels corresponding to the data samples 
        given by X
        
Returns:
    loss: The loss of the neural network on the samples
'''


def calculate_loss(model, X, y):
    # generate predictions
    y_hat = forward(model, X)
    labels = one_hot_encode(y)

    # calculate loss over all samples
    loss = -(1 / y.size) * np.sum(labels * np.log(y_hat))

    return loss


'''
Generates a predicted class based on the softmax provided by
the forward function. The output will be the class with the
highest confidence.

Params: 
    model: Neural network model with the same input dimension
        as the number of features given by X
    x: A single data sample represented as an array of features
    
Returns: 
    prediction: A numpy array containing the predicted class
'''


def predict(model, x):
    y_hat = forward(model, x)

    prediction = np.argmax(y_hat, axis=1)

    return np.array(prediction)


'''
Calculates a softmax prediction for the sample(s) given by X
using the neural network model. 

Params: 
    model: Neural network model with the same input dimension
        as the number of features given by X
    X: An array of data sample(s)
    
Returns:
    y_hat: An array of softmax predictions of classes for all 
        samples in X
'''


def forward(model, X):
    a = X @ model['W1'] + model['b1']
    h = np.tanh(a)
    z = h @ model['W2'] + model['b2']
    y_hat = softmax(z)

    return y_hat


'''
Builds and trains a neural network for num_passes on 
input X with labels y. The network architecture is
2 hidden layers of each size nn_hdim.

Params:
    X: An array of training data samples
    y: Corresponding labels for data samples contained
        in X
    nn_hdim: Size of each hidden layer
    num_passes: Number of iterations to train
    print_loss: A flag to print loss every thousand 
        iterations
    learning_rate: The learning rate of the neural
        network, used during the weight and bias
        update
    
Returns:
    model: A dictionary containing the weights and biases
        for the 2 model layers
'''


def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False, learning_rate=0.002):
    outputs = y.max() + 1

    # Randomly assign weights and biases
    model = {
        'W1': np.random.rand(X[0].size, nn_hdim),
        'W2': np.random.rand(nn_hdim, outputs),
        'b1': np.random.rand(nn_hdim),
        'b2': np.random.rand(outputs)
    }

    # modify labels to be one-hot encoded
    labels = one_hot_encode(y)

    for index in range(num_passes):
        # generate vector of predictions
        y_hat = forward(model, X)

        # compute gradients
        h = np.tanh(X @ model['W1'] + model['b1'])

        dLdy_hat = y_hat - labels
        dLda = (1 - h ** 2) * (dLdy_hat @ model['W2'].T)
        dLdW2 = h.T @ dLdy_hat
        dLdb2 = np.sum(dLdy_hat)
        dLdW1 = X.T @ dLda
        dLdb1 = np.sum(dLda)

        # update weights and biases
        model['W1'] -= learning_rate * dLdW1
        model['W2'] -= learning_rate * dLdW2
        model['b1'] -= learning_rate * dLdb1
        model['b2'] -= learning_rate * dLdb2

        # print loss every thousand iterations (if wanted)
        if print_loss and index % 1000 == 0:
            print(calculate_loss(model, X, y))

    return model


def build_model_691(X, y, nn_hdim, num_passes=20000, print_loss=False):
    return build_model(X, y, nn_hdim, num_passes, print_loss)


'''
Computes the softmax confidences for the given vector

Params: 
    z_vec: the vector to be converted to softmax
    
Returns:
    y_hat: vector of confidence for each class
'''


def softmax(z_vec):
    y_hat = np.exp(z_vec) / (np.sum(np.exp(z_vec), axis=1)).reshape(-1, 1)

    return y_hat


'''
One-hot encodes a list of labels to a list of 
one-hot labels

Params:
    vec: vector of labels, assumes all possible 
        classes are represented

Returns:
    labels: List of one-hot encoded labels
'''


def one_hot_encode(vec):
    labels = np.zeros((vec.size, vec.max() - vec.min() + 1))
    for index, label in enumerate(vec):
        labels[index, label] = 1

    return labels


'''
Plots the decision boundary for all data samples X
with labels y for the prediction function.

Params:
    pred_func: a function pointer to the 
        prediction function for the samples
    X: All data samples
    y: Corresponding labels for each data sample
    
Returns:
    None
'''


def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.get_cmap('Spectral'))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.get_cmap('Spectral'))

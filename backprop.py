import numpy as np

# features
x = np.array([[0.1, 0.2, 0.7]])

# labels
y = np.array([[1.0, 2.0, 2.0]])

# biases
b = np.array([[1.0, 1.0, 1.0]])

# init weights
w_1 = np.array([
    [0.1, 0.2, 0.3],
    [0.3, 0.2, 0.7],
    [0.4, 0.3, 0.9]])

w_2 = np.array([
    [0.2, 0.3, 0.5],
    [0.3, 0.5, 0.7],
    [0.6, 0.4, 0.8]
])

w_3 = np.array([
    [0.1, 0.4, 0.8],
    [0.3, 0.7, 0.2],
    [0.5, 0.2, 0.9]
])

# learning rate
l_r = 0.001


def h(x, w=None, b=None, derivative=False):
    if derivative:
        return x
    else:
        return x @ w + b


def relu(x, derivative=False):
    if derivative:
        return np.where(x <= 0, 0, 1)
    else:
        return x * (x > 0)


def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    else:
        return 1 / (1 + np.exp(-x))


def softmax(x, derivative=False):
    if derivative:
        return softmax(x) * (1 - softmax(x))
    else:
        return np.exp(x) / np.sum(np.exp(x))


def crossentropy(x, y, derivative=False):
    if derivative:
        return - (y / x)
    else:
        return - (y * np.log(x))


def b_crossentropy(x, y, derivative=False):
    if derivative:
        return - (y * 1 / x + (1 - y) * 1 / (1 - x))
    else:
        return - (y * np.log(x) + (1 - y) * np.log(1 - x))


# compute first hidden layer
h_1 = h(x, w_1, b)
z_1 = relu(h_1)

# compute second hidden layer
h_2 = h(z_1, w_2, b)
z_2 = sigmoid(h_2)

# compute output layer
h_3 = h(z_2, w_3, b)
z_3 = softmax(h_3)

# compute error
e_3 = b_crossentropy(z_3, y).sum()

# compute derivatives
d_e_z_3 = b_crossentropy(z_3, y, derivative=True)
d_z_h_3 = softmax(h_3, derivative=True)
d_h_w_3 = h(z_2, derivative=True)

d_e_z_2 = (d_z_h_3 * d_e_z_3 * w_2.T).sum(axis=1)
d_z_h_2 = sigmoid(h_2, derivative=True)
d_h_w_2 = h(z_1, derivative=True)

d_e_z_1 = (d_z_h_2 * d_e_z_2 * w_1.T).sum(axis=1)
d_z_h_1 = relu(h_1, derivative=True)
d_h_w_1 = h(x, derivative=True)

# compute delta using chain rule
d_e_w_3 = d_z_h_3 * d_e_z_3 * d_h_w_3.T
d_e_w_2 = d_e_z_2 * d_z_h_2 * d_h_w_2.T
d_e_w_1 = d_e_z_1 * d_z_h_1 * d_h_w_1.T

# compute new weights
delta_w_3 = w_3 - l_r * d_e_w_3
delta_w_2 = w_2 - l_r * d_e_w_2
delta_w_1 = w_1 - l_r * d_e_w_1

# assign new weights
w_3 = delta_w_3
w_2 = delta_w_2
w_1 = delta_w_1


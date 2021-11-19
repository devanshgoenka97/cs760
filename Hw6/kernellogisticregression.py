import numpy as np

def sigmoid(z):
    """
    Computes the sigmoid of the input vector which is given by 1/(1+e^-z)

    Params:
    z : The input vector

    Returns:
    The vector of element-wise sigmoid's applied to the input vector.
    """
    return 1.0/(1 + np.exp(-z))

class KernelLogisticRegression():
    """
    Implements Kernel Logistic Regression which evaluates P(y=1|x) using gradient descent.
    """

    def __init__(self, learning_rate = 1e-4, kernel = 'rbf', epochs = 100):
        """
        Initializes the Kernel Logistic Regression class with set of parameters.

        Params:
        learning_rate : The rate at which the gradients affect the weight updates.
        kernel : The kernel to use (polynomial or rbf).
        epochs : The number of iterations over the training set for which the model is trained.
        """
        self.learning_rate = learning_rate
        self.kernel = kernel
        self.epochs = epochs

    def compute_kernel(self, x, y, param = 5):
        """
        Computes k(x, y) for the given input.

        Params: 
        x : The first input to the kernel
        y : The second input to the kernel
        param : The param for the kernel (sigma for rbf kernel and degree for the polynomial kernel)

        Returns:
        A scalar value representing the computation of the kernel for the given inputs.
        """
        if self.kernel == 'rbf':
            return np.exp(-np.linalg.norm(x-y)**2 / (2 * (param ** 2)))
        else:
            return (1 + np.dot(x, y)) ** param

    def fit(self, X, y):
        """
        Trains the kernel logistic regression model using gradient descent over a number of epochs.

        Params:
        X : The set of (m, n) training points.
        y : The set of (m, ) true labels.
        """
        X = np.array(X)
        y = np.array(y)
        m, n = X.shape

        self.train = X
        self.train_length = m

        # Initializing alphas and bias to be zeros
        self.alphas = np.zeros(m)
        self.bias = 0

        self.gram = np.zeros((m, m))

        # Creating the gram matrix from the given input points
        for i in range(m):
            for j in range(m):
                self.gram[i, j] = self.compute_kernel(X[i], X[j])

        # Training loop.
        for epoch in range(self.epochs):
            for i in range(self.train_length):
                x_test = X[i].reshape(1, -1)

                # Predicting the values of the input given the current set of parameters
                predicted = self.predict(x_test)

                # Computing gradient of BCE loss of logistic regression
                gradient = (predicted - y[i]).item()

                for j in range(self.train_length):
                    # Updating each alpha using SGD
                    self.alphas[j] += self.learning_rate * gradient * self.gram[i, j]
        
                self.bias += self.learning_rate * gradient


    def predict(self, X):
        """
        Predicts the label of the given test points using the trained model.

        Params:
        X : The set of (m, n) test points.

        Returns:
        A list of size (m,) containing predicted [0, 1] labels for the test points.
        """
        m = X.shape[0]

        # Empty List to store predictions.
        pred_class = []

        # Calculating predictions.
        for i in range(m):
            z = 0
            for j in range(self.train_length):
                z += self.alphas[j] * self.compute_kernel(X[i], self.train[j])
            z += self.bias
            pred_class.append(sigmoid(z))
            
        return np.array(pred_class)

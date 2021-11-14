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

class LogisticRegression():
    """
    Implements standard logistic regression which evaluates P(y=1|x) using gradient descent.
    """

    def __init__(self, learning_rate = 1e-4, batch_size = 64, epochs = 100):
        """
        Initializes the Logistic Regression class with set of parameters.

        Params:
        learning_rate : The rate at which the gradients affect the weight updates.
        batch_size : The number of samples aftr which SGD is performed.
        epochs : The number of iterations over the training set for which the model is trained.
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def loss(self, y, y_hat):
        """
        Computes the binary cross entropy loss given by -[y log y^hat - (1-y) log(1-y^hat)]

        Params:
        y : The set of true labels
        y_hat : The set of predicted labels

        Returns:
        The averaged binary cross entropy loss over the minibatch.
        """
        loss = -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat))
        return loss

    def gradients(self, X, y, y_hat):
        """
        Computes the gradients necessary for backpropagation for the given data points and predicted labels.

        Params:
        X : The set of (m, n) training points.
        y : The set of (m,) true labels.
        y_hat : The set of (m,) predicted labels.

        Returns:
        dw : The gradient of the weights w.r.t the BCE loss function
        db : The graident of the bias w.r.t the BCE loss function
        """
        m = X.shape[0]
        
        # Gradient of loss w.r.t weights.
        dw = (1/m) * np.dot(X.T, (y_hat - y))
        
        # Gradient of loss w.r.t bias.
        db = (1/m) * np.sum((y_hat - y)) 
        
        return dw, db

    def fit(self, X, y):
        """
        Trains the logistic regression model using stochastic gradient descent over a number of epochs.

        Params:
        X : The set of (m, n) training points.
        y : The set of (m, ) true labels.

        Returns:
        w : The (n,) length trained weight vector
        b : The scalar value of the bias
        losses : A list of all losses at every epoch
        """
        X = np.array(X)
        y = np.array(y)
        m, n = X.shape
        
        # Initializing weights and bias to zeros.
        w = np.random.randn(n).reshape(n, 1)
        b = 0
        
        # Reshaping y.
        y = y.reshape(m, 1)
        
        # Empty list to store losses.
        losses = []
        
        # Training loop.
        for epoch in range(self.epochs):
            for i in range((m-1)//self.batch_size + 1):
                
                # Defining batches for SGD.
                start_i = i * self.batch_size
                end_i = start_i + self.batch_size
                xb = X[start_i:end_i]
                yb = y[start_i:end_i]
                
                # Calculating prediction.
                y_hat = sigmoid(np.dot(xb, w) + b)
                
                # Getting the gradients of loss w.r.t parameters.
                dw, db = self.gradients(xb, yb, y_hat)
                
                # Updating the parameters.
                w -= self.learning_rate * dw
                b -= self.epochs * db
            
            # Calculating loss and appending it in the list.
            l = self.loss(y, sigmoid(np.dot(X, w) + b))
            losses.append(l)
            
        # Storing weights, bias and losses.
        self.w = w
        self.b = b

        return w, b, losses

    def predict(self, X):
        """
        Predicts the label of the given test points using the trained model.

        Params:
        X : The set of (m, n) test points.

        Returns:
        A list of size (m,) containing predicted [0, 1] labels for the test points.
        """
        
        # Calculating predictions.
        preds = sigmoid(np.dot(X, self.w) + self.b)
        
        # Empty List to store predictions.
        pred_class = []

        # Clamp prediction to [0, 1] using 0.5 as a threshold.
        pred_class = [1 if i > 0.5 else 0 for i in preds]
        
        return np.array(pred_class)

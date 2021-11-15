import numpy as np
import cvxopt
import cvxopt.solvers
             
def linear(x1, x2):
    """
    The linear kernel is just the dot product of the two input vectors.

    Params:
    x1 : A vector of size (m,)
    x2 : A vector of size (m,)

    Returns:
    A single scalar representing the inner product of the two vectors.
    """
    return np.dot(x1, x2)

def polynomial(x, y, p=3):
    """
    The polyonomial kernel.

    Params:
    x : A vector of size (m,)
    y : A vector of size (m,)
    p : The degree of the polynomial. Defauls to 3.

    Returns:
    A single scalar representing the polynomial's inner product.
    """
    return (1 + np.dot(x, y)) ** p

def rbf(x, y, sigma=5.0):
    """
    Computes the radial basis function or gaussian kernel.

    Params:
    x : A vector of size (m,)
    y : A vector of size (m,)
    sigma : A float value representing the variance term in the gaussian kernel. Scaling sigma changes the shape of the decision boundary. Defaults to 5.

    Returns:
    A single scalar representing the kernel's value.
    """
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

class SVM():
    """
    Implements the hard-margin and soft-margin SVM using quadratic programming and the kernel trick.
    """

    def __init__(self, kernel='linear', C=None, degree=2, sigma=5):
        """
        Initializes the SVM class with given parameters. 

        Params:
        kernel : A string which determines the type of kernel to use for the inner product computation. Defaults to linear.
        C : A real number which defines the soft-margin scale of the slack variables in the SVM. Defaults to None, which implies the hard-margin SVM.
        """
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.sigma = sigma
        if self.C is not None: 
            self.C = float(self.C)

    def fit(self, X, y):
        """
        Trains the maximal margin classifier over the given data using quadratic programming.
        Calculates the dual of the Lagrangian and stores the resulting support vectors along with the weights and bias of the hyperplane.

        Params:
        X : The set of (m, n) training points
        y : The set of (m, ) labels for the training points in [-1, 1]
        """
        m, n = X.shape

        # Constructing the gram matrix
        K = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                # Kernel trick.
                if self.kernel == 'linear':
                    K[i, j] = linear(X[i], X[j])
                if self.kernel=='rbf':
                    K[i, j] = rbf(X[i], X[j], self.sigma)
                    self.C = None
                if self.kernel == 'polynomial':
                    K[i, j] = polynomial(X[i], X[j], self.degree)
        
        # Adding small noise to ensure it is positive semi-definite
        K = K + (1e-4) * np.eye(m)
        
        # Converting y to a float type as cvxopt expects 'd' matrix
        y = y * 1.

        # Framing the constrained optimization in cvxopt
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones((m, 1)))
        A = cvxopt.matrix(y, (1, m))
        b = cvxopt.matrix(np.zeros(1))
        
        # If C is not defined, use the hard margin SVM
        if self.C is None or self.C==0:
            G = cvxopt.matrix(-np.eye(m))
            h = cvxopt.matrix(np.zeros(m))
        else:
            # Soft margin SVM with slack variables tempered by C
            tmp1 = -np.eye(m)
            tmp2 = np.eye(m)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(m)
            tmp2 = np.ones(m) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # Setting options
        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['abstol'] = 1e-7
        cvxopt.solvers.options['reltol'] = 1e-6
        cvxopt.solvers.options['feastol'] = 1e-7

        #Run solver
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        alphas = np.ravel(solution['x']) 

        # Support vectors have non zero lagrange multipliers
        sv = alphas > 1e-5
        ind = np.arange(len(alphas))[sv]
        self.alphas = alphas[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        # Bias (For linear it is the intercept):
        self.b = 0
        for p in range(len(self.alphas)):
            # For all support vectors:
            self.b += self.sv_y[p]
            self.b -= np.sum(self.alphas * self.sv_y * K[ind[p], sv])
        self.b = self.b / len(self.alphas)

        # Weight vector
        if self.kernel == 'linear':
            self.w = np.zeros(n)
            for q in range(len(self.alphas)):
                self.w += self.alphas[q] * self.sv_y[q] * self.sv[q]
        else:
            self.w = None
        
        print('Found the optimal solution')

        if self.kernel == 'linear':
            print(f'w = {self.w}')
            print(f'b = {self.b}')

    def project(self, X):
        """
        Projects the set of test points to the feature space using y(w.Tx + b).
        If there is a non-linear kernel then the projection is done using the lagrange multipliers.

        Params:
        X : The set of (m, n) test points.

        Returns:
        A single scalar representing the predicted projection of the test points on the feature space.
        """
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.alphas, self.sv_y, self.sv):
                    if self.kernel == 'linear':
                        s += a * sv_y * linear(X[i], sv)
                    if self.kernel=='rbf':
                        s += a * sv_y * rbf(X[i], sv, self.sigma)
                    if self.kernel == 'polynomial':
                        s += a * sv_y * polynomial(X[i], sv, self.degree)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        """
        Predicts the labels for the given set of test points from the trained classifier.

        Params:
        X : The set of (m, n) test points

        Returns:
        The set of (m, ) predicted labels from the classifier
        """
        return np.sign(self.project(X))

from cvxopt import uniform
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from logisticregression import LogisticRegression
from svm import SVM

import matplotlib.pyplot as pl
import numpy as np

COLORS = ['red', 'blue']

def plot_separator(ax, w, b):
        slope = -w[0] / w[1]
        intercept = -b / w[1]
        ax.autoscale(False)
        x_vals = np.array(ax.get_xlim())
        y_vals = intercept + (slope * x_vals)
        ax.plot(x_vals, y_vals, 'k-')

def plot_data_with_labels(x, y, ax):
    # Scattering training points on axis
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    unique = np.unique(y)
    for li in range(len(unique)):
        x_sub = x[y == unique[li]]
        ax.scatter(x_sub[:, 0], x_sub[:, 1], c = COLORS[li], zorder = 1, s = 8)

def generate_gaussian_2d_data(mu = 2.5):
    # Sample from 2 2D Gaussian distributions
    mean1 = np.array([-mu, 0])
    mean2 = np.array([mu, 0])
    cov = np.eye(2)

    X1 = np.random.multivariate_normal(mean1, cov, 750)
    y1 = np.ones(len(X1))

    X2 = np.random.multivariate_normal(mean2, cov, 750)
    y2 = np.ones(len(X2)) * -1

    return X1, y1, X2, y2

def plot_margin(X1_train, X2_train, clf, ax):
    # Plot training points
    x = np.vstack((X1_train, X2_train))
    y = np.hstack((np.ones(X1_train.shape[0]), np.zeros(X2_train.shape[0]) ))
    plot_data_with_labels(x, y, ax)

    # Plot decision boundary
    w, bias = clf.w, clf.b
    plot_separator(ax, w, bias)

def plot_contour(X1_train, X2_train, clf, ax):
    # Plot training points
    x = np.vstack((X1_train, X2_train))
    y = np.hstack((np.ones(X1_train.shape[0]), np.zeros(X2_train.shape[0])))
    plot_data_with_labels(x, y, ax)

    # Plot decision contour
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, zorder = -1, cmap = 'gray')

def test_linear():
    # Generating synthetic dataset 1 : Sampling from 2 multivariate Gaussians
    X1, y1, X2, y2 = generate_gaussian_2d_data(mu = 2.5)

    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))

    # Splitting into train, test and val sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=int(1250), random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=int(1000), random_state=42)

    # Performing classification with Linear SVM
    clf = SVM(kernel='linear')
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)

    # Calculating test accuracy
    print("Linear SVM Classifier")
    print("----------------------")
    print(f"Test accuracy : {(correct/len(y_predict))*100.}")
    print(f"{correct} out of {len(y_predict)} correctly classified")
    print("")

    # Plot decision boundary
    fig, ax = pl.subplots()
    plot_margin(X_train[y_train==1], X_train[y_train==-1], clf, ax)
    ax.set_title("Decision boundary for Linear SVM")
    pl.figure(figsize=(8, 6))
    pl.axis("tight")
    fig.show()
    input("Close the figure and press a key to continue")

    # Converting labels from [-1, 1] to [0, 1] for other classifiers
    y_train[y_train==-1] = 0
    y_test[y_test==-1] = 0

    # Performing classification with Logistic Regression
    clf = LogisticRegression(batch_size=20, epochs=1500, learning_rate=0.03)
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)

    # Calculating test accuracy
    print("Logistic Regression Classifier")
    print("----------------------")
    print(f"Test accuracy : {(correct/len(y_predict))*100.}")
    print(f"{correct} out of {len(y_predict)} correctly classified")
    print("")

    # Plot decision boundary
    fig, ax = pl.subplots()
    plot_margin(X_train[y_train==1], X_train[y_train==0], clf, ax)
    ax.set_title("Decision boundary for Logistic Regression")
    pl.figure(figsize=(8, 6))
    pl.axis("tight")
    fig.show()
    input("Close the figure and press a key to continue")

    # Performing classification with K nearest neighbors
    clf = KNeighborsClassifier(n_neighbors=15)
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)

    # Calculating test accuracy
    print("K Nearest Neighbors Classifier")
    print("----------------------")
    print(f"Test accuracy : {(correct/len(y_predict))*100.}")
    print(f"{correct} out of {len(y_predict)} correctly classified")
    print("")

    # Plot the decision boundary.
    fig, ax = pl.subplots()
    plot_contour(X_train[y_train==1], X_train[y_train==0], clf, ax)
    ax.set_title("Decision boundary for K Nearest Neighbors")
    pl.figure(figsize=(8, 6))
    fig.show()
    input("Close the figure and press a key to continue")

    # Performing classification with K nearest neighbors
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)

    # Calculating test accuracy
    print("Gaussian Naive Bayes Classifier")
    print("----------------------")
    print(f"Test accuracy : {(correct/len(y_predict))*100.}")
    print(f"{correct} out of {len(y_predict)} correctly classified")
    print("")
    
    # Plot the decision boundary.
    fig, ax = pl.subplots()
    plot_contour(X_train[y_train==1], X_train[y_train==0], clf, ax)
    ax.set_title("Decision boundary for Gaussian Naive Bayes")
    pl.figure(figsize=(8, 6))
    fig.show()
    input("Close the figure and press a key to continue")

    # Creating varying gaussians
    means = np.linspace(1., 2.4, num=8)
    accuracies = {'Linear SVM': [], 'Logistic Regression': [], 'K Nearest Neighbors': [], 'Gaussian Naive Bayes': []}
    for mean in means:
        print(f"For Gaussian with mean : {mean}")

        X1, y1, X2, y2 = generate_gaussian_2d_data(mu = mean)
        X = np.vstack((X1, X2))
        y = np.hstack((y1, y2))
        
        # Splitting into train, test and val sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=int(1250), random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=int(1000), random_state=42)

        # Performing classification with Linear SVM
        clf = SVM(kernel='linear')
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)

        # Calculating test accuracy
        print("Linear SVM Classifier")
        print("----------------------")
        print(f"Test accuracy : {(correct/len(y_predict))*100.}")
        print(f"{correct} out of {len(y_predict)} correctly classified")
        print("")
        accuracies['Linear SVM'].append((correct/len(y_predict))*100.)

        # Converting labels from [-1, 1] to [0, 1] for other classifiers
        y_train[y_train==-1] = 0
        y_test[y_test==-1] = 0

        # Performing classification with Logistic Regression
        clf = LogisticRegression(batch_size=20, epochs=1500, learning_rate=0.03)
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)

        # Calculating test accuracy
        print("Logistic Regression Classifier")
        print("----------------------")
        print(f"Test accuracy : {(correct/len(y_predict))*100.}")
        print(f"{correct} out of {len(y_predict)} correctly classified")
        print("")
        accuracies['Logistic Regression'].append((correct/len(y_predict))*100.)

        # Performing classification with K nearest neighbors
        clf = KNeighborsClassifier(n_neighbors=15)
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)

        # Calculating test accuracy
        print("K Nearest Neighbors Classifier")
        print("----------------------")
        print(f"Test accuracy : {(correct/len(y_predict))*100.}")
        print(f"{correct} out of {len(y_predict)} correctly classified")
        print("")
        accuracies['K Nearest Neighbors'].append((correct/len(y_predict))*100.)

        # Performing classification with K nearest neighbors
        clf = GaussianNB()
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)

        # Calculating test accuracy
        print("Gaussian Naive Bayes Classifier")
        print("----------------------")
        print(f"Test accuracy : {(correct/len(y_predict))*100.}")
        print(f"{correct} out of {len(y_predict)} correctly classified")
        print("")
        accuracies['Gaussian Naive Bayes'].append((correct/len(y_predict))*100.)
    
    # Plotting the mean vs accuracy graph for all classifiers
    for classifier in accuracies.keys():
        fig, ax = pl.subplots()
        accs = accuracies[classifier]
        ax.plot(means, accs)
        ax.set_xlim(1, 2.4)
        ax.set_title(f"Test Accuracy vs Mean for {classifier}")
        ax.set_xlabel("Mean of the 2D Gaussians")
        ax.set_ylabel("Test Accuracy")
        fig.show()
        input("Close the figure and press a key to continue")


def test_circles():

    X, y = make_circles(n_samples=1500, random_state=42)
    y[y==0] = -1

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=int(1250), random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=int(1000), random_state=42)

    clf = SVM(kernel='rbf')
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))

    #plot_margin(X_train[y_train==1], X_train[y_train==-1], clf)


if __name__ == "__main__":
    test_linear()
    #test_circles()
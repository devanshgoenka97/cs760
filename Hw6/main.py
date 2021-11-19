from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_circles, load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from kernellogisticregression import KernelLogisticRegression
from logisticregression import LogisticRegression
from svm import SVM
from l1_svm import L1SVM
from sklearn.feature_selection import SelectFromModel
from neuralnetwork import NeuralNetwork

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

def evaluate_gaussian_dataset():
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

def evaluate_circles_dataset():
    # Generating 2D circles from dataset
    X, y = make_circles(n_samples=1500, random_state=42)
    y[y==0] = -1

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

    # Performing classification with Polynomial Kernel SVM
    clf = SVM(kernel='polynomial')
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)

    # Calculating test accuracy
    print("Polynomial Kernel SVM Classifier")
    print("----------------------")
    print(f"Test accuracy : {(correct/len(y_predict))*100.}")
    print(f"{correct} out of {len(y_predict)} correctly classified")
    print("")

    # Plot the decision boundary.
    fig, ax = pl.subplots()
    plot_contour(X_train[y_train==1], X_train[y_train==-1], clf, ax)
    ax.set_title("Decision boundary for Polynomial Kernel SVM")
    pl.figure(figsize=(8, 6))
    fig.show()
    input("Close the figure and press a key to continue")

    # Performing classification with RBF Kernel SVM
    clf = SVM(kernel='rbf')
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)

    # Calculating test accuracy
    print("RBF Kernel SVM Classifier")
    print("----------------------")
    print(f"Test accuracy : {(correct/len(y_predict))*100.}")
    print(f"{correct} out of {len(y_predict)} correctly classified")
    print("")

    # Plot the decision boundary.
    fig, ax = pl.subplots()
    plot_contour(X_train[y_train==1], X_train[y_train==-1], clf, ax)
    ax.set_title("Decision boundary for RBF Kernel SVM")
    pl.figure(figsize=(8, 6))
    fig.show()
    input("Close the figure and press a key to continue")

    # Using labels [0,1] for other classifiers
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

    # Performing classification with Kernel Logistic Regression
    clf = KernelLogisticRegression(epochs=100, learning_rate=0.03)
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    # Clamp prediction to [0, 1] using 0.5 as a threshold.
    y_predict = np.array([1 if i > 0.5 else 0 for i in y_predict])
    correct = np.sum(y_predict == y_test)

    # Calculating test accuracy
    print("Kernel Logistic Regression Classifier")
    print("----------------------")
    print(f"Test accuracy : {(correct/len(y_predict))*100.}")
    print(f"{correct} out of {len(y_predict)} correctly classified")
    print("")

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

    # Performing classification with Neural Network classifier
    clf = NeuralNetwork(epochs=100, learning_rate=0.3)
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)

    # Calculating test accuracy
    print("Neural Network Classifier")
    print("----------------------")
    print(f"Test accuracy : {(correct/len(y_predict))*100.}")
    print(f"{correct} out of {len(y_predict)} correctly classified")
    print("")

    # Plot the decision boundary.
    fig, ax = pl.subplots()
    plot_contour(X_train[y_train==1], X_train[y_train==0], clf, ax)
    ax.set_title("Decision boundary for Neural Network classifier")
    pl.figure(figsize=(8, 6))
    fig.show()
    input("Close the figure and press a key to continue")

def evaluate_breast_cancer_dataset():
    # Using the Wisconsin Breast Cancer dataset
    X, y = load_breast_cancer(return_X_y=True)

    # Splitting into train, test and val sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    # Normalizing the features in the datasets
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

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

    # Performing classification with K nearest neighbors
    clf = KNeighborsClassifier(n_neighbors=15)
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)

    # Performing classification with Neural Network classifier
    clf = NeuralNetwork(input_size=X_train.shape[1], epochs=100, learning_rate=0.3)
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)

    # Calculating test accuracy
    print("Neural Network Classifier")
    print("----------------------")
    print(f"Test accuracy : {(correct/len(y_predict))*100.}")
    print(f"{correct} out of {len(y_predict)} correctly classified")
    print("")

    # Calculating test accuracy
    print("K Nearest Neighbors Classifier")
    print("----------------------")
    print(f"Test accuracy : {(correct/len(y_predict))*100.}")
    print(f"{correct} out of {len(y_predict)} correctly classified")
    print("")

    # Performing classification with Gaussian Naive Bayes
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

    # Converting to [1, -1] labels for the SVM classifiers
    y_train[y_train==0] = -1
    y_test[y_test==0] = -1

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

    # Performing classification with Polynomial Kernel SVM
    clf = SVM(kernel='polynomial', C=0.1)
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)

    # Calculating test accuracy
    print("Polynomial SVM Classifier")
    print("----------------------")
    print(f"Test accuracy : {(correct/len(y_predict))*100.}")
    print(f"{correct} out of {len(y_predict)} correctly classified")
    print("")

    # Performing classification with Gaussian Kernel SVM
    clf = SVM(kernel='rbf')
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)

    # Calculating test accuracy
    print("RBF Kernel SVM Classifier")
    print("----------------------")
    print(f"Test accuracy : {(correct/len(y_predict))*100.}")
    print(f"{correct} out of {len(y_predict)} correctly classified")
    print("")

    # Performing classification with L1 penalty SVM classifier
    clf = L1SVM(input_size=X_train.shape[1], epochs=500, learning_rate=0.03, lamda=0.5)
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)

    # Calculating test accuracy
    print("L1 SVM Classifier")
    print("----------------------")
    print(f"Test accuracy : {(correct/len(y_predict))*100.}")
    print(f"{correct} out of {len(y_predict)} correctly classified")
    print("")


if __name__ == "__main__":
    evaluate_gaussian_dataset()
    #evaluate_circles_dataset()
    #evaluate_breast_cancer_dataset()
import math
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

if __name__ == '__main__':
    df = pd.read_csv('data/emails.csv', sep=",")
    # Dropping the 1st column because it contains index
    df = df.drop(columns=df.columns[0])

    # Creating the individual splits
    fold1_train = df.iloc[:1000, :]
    fold1_test = df.iloc[1000:, :]

    fold2_train = df.iloc[1000:2000, :]
    fold2_test = pd.concat([df.iloc[:1000, :], df.iloc[2000:, :]])

    fold3_train = df.iloc[2000:3000, :]
    fold3_test = fold2_test = pd.concat([df.iloc[:2000, :], df.iloc[3000:, :]])

    fold4_train = df.iloc[3000:4000, :]
    fold4_test = fold2_test = pd.concat([df.iloc[:3000, :], df.iloc[4000:, :]])

    fold5_train = df.iloc[4000:5000, :]
    fold5_test = df.iloc[:3000, :]

    foldmain_train = df.iloc[:4000, :]
    foldmain_test = df.iloc[4000:5000, :]

    # neigh = KNeighborsClassifier(n_neighbors=5)
    # neigh.fit(foldmain_train.iloc[:, :-1], foldmain_train.iloc[:, -1])
    # y_pred = neigh.predict(foldmain_test.iloc[:, :-1])
    # fpr, tpr, threshold = roc_curve(foldmain_test.iloc[:, -1], y_pred)

    # lr = LogisticRegression(max_iter=1000)
    # lr.fit(foldmain_train.iloc[:, :-1], foldmain_train.iloc[:,-1])
    # predicted_classes = lr.predict(foldmain_test.iloc[:, :-1])

    # plt.title('Receiver Operating Characteristic')
    # plt.plot(fpr, tpr, 'b', color="violet", label="kNN : AUC = 0.770")
    # fpr, tpr, threshold = roc_curve(foldmain_test.iloc[:, -1], predicted_classes)
    # plt.plot(fpr, tpr, 'b', color="purple", label="Logistic Regression : AUC = 0.936")
    # plt.legend(loc = 'lower right')
    # plt.plot([0, 1], [0, 1],'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.title('ROC Curve of kNN and Logistic Regression')
    # plt.show()


    for i in [1,3,5,7,10]:
        var = 0
        neigh = KNeighborsClassifier(n_neighbors=i)
        neigh.fit(fold1_train.iloc[:, :-1], fold1_train.iloc[:, -1])
        y_pred = neigh.predict(fold1_test.iloc[:, :-1])
        var = var + accuracy_score(fold1_test.iloc[:, -1], y_pred)
        #print("Accuracy : 1 : " + str(accuracy_score(fold1_test.iloc[:, -1], y_pred)))
        #print("Precision : 1 : " + str(precision_score(fold1_test.iloc[:, -1], y_pred, average='binary')))
        #print("Recall : 1 : " + str(recall_score(fold1_test.iloc[:, -1], y_pred, average='binary')))

        neigh = KNeighborsClassifier(n_neighbors=i)
        neigh.fit(fold2_train.iloc[:, :-1], fold2_train.iloc[:, -1])
        y_pred = neigh.predict(fold2_test.iloc[:, :-1])
        var = var + accuracy_score(fold2_test.iloc[:, -1], y_pred)
        #print("Accuracy : 2 : " + str(accuracy_score(fold2_test.iloc[:, -1], y_pred)))
        #print("Precision : 2 : " + str(precision_score(fold2_test.iloc[:, -1], y_pred, average='binary')))
        #print("Recall : 2 : " + str(recall_score(fold2_test.iloc[:, -1], y_pred, average='binary')))

        neigh = KNeighborsClassifier(n_neighbors=i)
        neigh.fit(fold3_train.iloc[:, :-1], fold3_train.iloc[:, -1])
        y_pred = neigh.predict(fold3_test.iloc[:, :-1])
        var = var + accuracy_score(fold3_test.iloc[:, -1], y_pred)
        #print("Accuracy : 3 : " + str(accuracy_score(fold3_test.iloc[:, -1], y_pred)))
        #print("Precision : 3 : " + str(precision_score(fold3_test.iloc[:, -1], y_pred, average='binary')))
        #print("Recall : 3 : " + str(recall_score(fold3_test.iloc[:, -1], y_pred, average='binary')))

        neigh = KNeighborsClassifier(n_neighbors=i)
        neigh.fit(fold4_train.iloc[:, :-1], fold4_train.iloc[:, -1])
        y_pred = neigh.predict(fold4_test.iloc[:, :-1])
        var = var + accuracy_score(fold4_test.iloc[:, -1], y_pred)
        #print("Accuracy : 4 : " + str(accuracy_score(fold4_test.iloc[:, -1], y_pred)))
        #print("Precision : 4 : " + str(precision_score(fold4_test.iloc[:, -1], y_pred, average='binary')))
        #print("Recall : 4 : " + str(recall_score(fold4_test.iloc[:, -1], y_pred, average='binary')))

        neigh = KNeighborsClassifier(n_neighbors=i)
        neigh.fit(fold5_train.iloc[:, :-1], fold5_train.iloc[:, -1])
        y_pred = neigh.predict(fold5_test.iloc[:, :-1])
        var = var + accuracy_score(fold5_test.iloc[:, -1], y_pred)
        #print("Accuracy : 5 : " + str(accuracy_score(fold5_test.iloc[:, -1], y_pred)))
        #print("Precision : 5 : " + str(precision_score(fold5_test.iloc[:, -1], y_pred, average='binary')))
        #print("Recall : 5 : " + str(recall_score(fold5_test.iloc[:, -1], y_pred, average='binary')))
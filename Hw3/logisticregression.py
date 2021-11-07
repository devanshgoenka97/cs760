from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

def loss(y, y_hat):
    loss = -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat))
    return loss

def gradients(X, y, y_hat):
    
    # X --> Input.
    # y --> true/target value.
    # y_hat --> hypothesis/predictions.
    # w --> weights (parameter).
    # b --> bias (parameter).
    
    # m-> number of training examples.
    m = X.shape[0]
    
    # Gradient of loss w.r.t weights.
    dw = (1/m)*np.dot(X.T, (y_hat - y))
    
    # Gradient of loss w.r.t bias.
    db = (1/m)*np.sum((y_hat - y)) 
    
    return dw, db

def normalize(X):
    
    # X --> Input.
    
    # m-> number of training examples
    # n-> number of features 
    m, n = X.shape
    
    # Normalizing all the n features of X.
    for i in range(n):
        X = (X - X.mean(axis=0))/X.std(axis=0)
        
    return X

def train(X, y, bs, epochs, lr):
    
    # X --> Input.
    # y --> true/target value.
    # bs --> Batch Size.
    # epochs --> Number of iterations.
    # lr --> Learning rate.
        
    # m-> number of training examples
    # n-> number of features 
    X = np.array(X)
    y = np.array(y)
    m, n = X.shape
    
    # Initializing weights and bias to zeros.
    w = np.zeros((n,1))
    b = 0
    
    # Reshaping y.
    y = y.reshape(m,1)

    # Normalizing the inputs.
    #x = normalize(X)
    
    # Empty list to store losses.
    losses = []
    
    # Training loop.
    for epoch in range(epochs):
        for i in range((m-1)//bs + 1):
            
            # Defining batches. SGD.
            start_i = i*bs
            end_i = start_i + bs
            xb = X[start_i:end_i]
            yb = y[start_i:end_i]
            
            # Calculating hypothesis/prediction.
            y_hat = sigmoid(np.dot(xb, w) + b)
            
            # Getting the gradients of loss w.r.t parameters.
            dw, db = gradients(xb, yb, y_hat)
            
            # Updating the parameters.
            w -= lr*dw
            b -= lr*db
        
        # Calculating loss and appending it in the list.
        l = loss(y, sigmoid(np.dot(X, w) + b))
        losses.append(l)
        
    # returning weights, bias and losses(List).
    return w, b, losses

def predict(X):
    
    # X --> Input.
    
    # Normalizing the inputs.
    #x = normalize(X)
    
    # Calculating presictions/y_hat.
    preds = sigmoid(np.dot(X, w) + b)
    
    # Empty List to store predictions.
    pred_class = []
    # if y_hat >= 0.5 --> round up to 1
    # if y_hat < 0.5 --> round up to 1
    pred_class = [1 if i > 0.5 else 0 for i in preds]
    
    return np.array(pred_class)

def accuracy(y, y_hat):
    accuracy = np.sum(y == y_hat) / len(y)
    return accuracy

df = pd.read_csv('data/emails.csv', sep=",")
df = df.drop(columns=df.columns[0])

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

model1fold = LogisticRegression(max_iter=1000)
w, b, l = train(fold1_train.iloc[:, :-1], fold1_train.iloc[:,-1], bs=100, epochs=1000, lr=0.01)
predicted_c = predict(fold1_test.iloc[:, :-1])
print("Custom 1fold"+str(accuracy(np.array(fold1_test.iloc[:, -1]),predicted_c)))

model1fold.fit(fold1_train.iloc[:, :-1], fold1_train.iloc[:,-1])
predicted_classes = model1fold.predict(fold1_test.iloc[:, :-1])

acc = accuracy_score(np.array(fold1_test.iloc[:, -1]),predicted_classes)
print("1 F Precision " + str(precision_score(np.array(fold1_test.iloc[:, -1]),predicted_classes, average='binary')))
print("1 F Recall " + str(recall_score(np.array(fold1_test.iloc[:, -1]),predicted_classes, average='binary')))
print("SKLearn 1fold"+str(acc))

parameters = model1fold.coef_

model2fold = LogisticRegression(max_iter=1000)
model2fold.fit(fold2_train.iloc[:, :-1], fold2_train.iloc[:,-1])
predicted_classes = model2fold.predict(fold2_test.iloc[:, :-1])
acc = accuracy_score(np.array(fold2_test.iloc[:, -1]),predicted_classes)
print("SKLearn 2fold"+str(acc))
print("2 F Precision " + str(precision_score(np.array(fold2_test.iloc[:, -1]),predicted_classes, average='binary')))
print("2 F Recall " + str(recall_score(np.array(fold2_test.iloc[:, -1]),predicted_classes, average='binary')))

w, b, l = train(fold2_train.iloc[:, :-1], fold2_train.iloc[:,-1], bs=100, epochs=1000, lr=0.01)
predicted_c = predict(fold2_test.iloc[:, :-1])

print(accuracy(np.array(fold2_test.iloc[:, -1]),predicted_c))


parameters = model2fold.coef_

model3fold = LogisticRegression(max_iter=1000)
model3fold.fit(fold3_train.iloc[:, :-1], fold3_train.iloc[:,-1])
predicted_classes = model3fold.predict(fold3_test.iloc[:, :-1])
acc = accuracy_score(np.array(fold3_test.iloc[:, -1]),predicted_classes)
print("SKLearn 3fold"+str(acc))
print("3 F Precision " + str(precision_score(np.array(fold3_test.iloc[:, -1]),predicted_classes, average='binary')))
print("3 F Recall " + str(recall_score(np.array(fold3_test.iloc[:, -1]),predicted_classes, average='binary')))

w, b, l = train(fold3_train.iloc[:, :-1], fold3_train.iloc[:,-1], bs=100, epochs=1000, lr=0.01)
predicted_c = predict(fold3_test.iloc[:, :-1])
print("Custom 3fold"+str(accuracy(np.array(fold3_test.iloc[:, -1]),predicted_c)))

parameters = model3fold.coef_

model4fold = LogisticRegression(max_iter=1000)
model4fold.fit(fold4_train.iloc[:, :-1], fold4_train.iloc[:,-1])
predicted_classes = model4fold.predict(fold4_test.iloc[:, :-1])
acc = accuracy_score(np.array(fold4_test.iloc[:, -1]),predicted_classes)
print("SKLearn 4fold"+str(acc))
print("4 F Precision " + str(precision_score(np.array(fold4_test.iloc[:, -1]),predicted_classes, average='binary')))
print("4 F Recall " + str(recall_score(np.array(fold4_test.iloc[:, -1]),predicted_classes, average='binary')))

w, b, l = train(fold4_train.iloc[:, :-1], fold4_train.iloc[:,-1], bs=100, epochs=1000, lr=0.01)
predicted_c = predict(fold4_test.iloc[:, :-1])
print("Custom 4fold"+str(accuracy(np.array(fold4_test.iloc[:, -1]),predicted_c)))

parameters = model4fold.coef_

model5fold = LogisticRegression(max_iter=1000)
model5fold.fit(fold5_train.iloc[:, :-1], fold5_train.iloc[:,-1])
predicted_classes = model5fold.predict(fold5_test.iloc[:, :-1])
acc = accuracy_score(np.array(fold5_test.iloc[:, -1]),predicted_classes)
print("SKLearn 5fold"+str(acc))
print("5 F Precision " + str(precision_score(np.array(fold5_test.iloc[:, -1]),predicted_classes, average='binary')))
print("5 F Recall " + str(recall_score(np.array(fold5_test.iloc[:, -1]),predicted_classes, average='binary')))

w, b, l = train(fold5_train.iloc[:, :-1], fold5_train.iloc[:,-1], bs=100, epochs=1000, lr=0.01)
predicted_c = predict(fold5_test.iloc[:, :-1])
print("Custom 5fold"+str(accuracy(np.array(fold5_test.iloc[:, -1]),predicted_c)))

parameters = model5fold.coef_

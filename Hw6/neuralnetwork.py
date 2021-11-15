import numpy as np
from torch import nn
import torch

class NeuralNetwork():
    def __init__(self, input_size = 2, output_size = 1, epochs = 100, batch_size = 20, learning_rate = 0.001):
        self.input_size = input_size
        self.hidden_size = 300
        self.epochs = epochs
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.criterion = nn.BCEWithLogitsLoss()
        # Build a simple 2-layer feed forward network as described
        self.model = nn.Sequential(nn.Linear(self.input_size, self.hidden_size, bias=False),
                      nn.Sigmoid(),
                      nn.Linear(self.hidden_size, self.output_size, bias=False))
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.0)

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        m, n = X.shape
        for i in range(self.epochs):
            running_loss = 0
            for j in range((m-1)//self.batch_size + 1):

                # Defining batches for SGD.
                start = j * self.batch_size
                end = start + self.batch_size
                xb = torch.tensor(X[start:end])
                yb = torch.tensor(y[start:end]).float()

                # Training pass
                self.optimizer.zero_grad()
                
                self.model.to(torch.double)
                output = self.model(xb).squeeze()
                loss = self.criterion(output, yb)
                
                # This is where the model learns by backpropagating
                loss.backward()
                
                # And optimizes its weights here
                self.optimizer.step()
                
                running_loss += loss.item()
            else:
                continue
                #print("Epoch {0}, Training loss: {1}".format(i, running_loss/len(xb)))
    
    def predict(self, X):
        X = np.array(X)
        m, n = X.shape
        all_predictions = np.zeros(m)
        for i in range((m-1)//self.batch_size + 1):
            # Defining batches.
            start = i * self.batch_size
            end = start + self.batch_size
            xb = torch.tensor(X[start:end])

            # Turn off gradients for forward pass
            with torch.no_grad():
                logits = self.model(xb)
                # Transforming logit to [0,1] label and then rounding off to predict binary class
                predictions = torch.round(torch.sigmoid(logits)).squeeze().detach().numpy()
                all_predictions[start:end] = predictions

        return np.array(all_predictions)
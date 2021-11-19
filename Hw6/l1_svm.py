import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch

# Implementing the hinge loss of the SVM as a class, to use the autograd feature of PyTorch.
class SVM_HingeLoss(nn.modules.Module): 
    def __init__(self, batch_size):
        super(SVM_HingeLoss,self).__init__()
        self.batch_size = batch_size

    def forward(self, outputs, labels):
        # Standard SVM hinge loss: (1/N) \sum [max(0, 1-y(w^t x + b))]
        return torch.sum(torch.clamp(1 - outputs.t()*labels, min=0))/self.batch_size

class L1SVM():
    """
    Implements an SVM with L1 norm penalty to induce sparsity.
    """

    def __init__(self, input_size = 2, output_size = 1, epochs = 100, batch_size = 20, learning_rate = 0.001, lamda = 0.1):
        """
        Initializes the SVM classifier

        Params:
        input_size : The number of features in the input space (Default 2).
        output_size : The number of classes to be predicted (Default 1).
        epochs : The number of epochs to train for.
        batch_size : The size of the minibatch for SGD.
        learning_rate : The rate at which the gradients affect the weight updation.
        """
        self.input_size = input_size
        self.epochs = epochs
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # Build a simple linear layer to compute w^t x + b
        self.model = nn.Linear(self.input_size, self.output_size, bias=True)
        # Using the defined hinge loss for SVM
        self.criterion = SVM_HingeLoss(batch_size = self.batch_size)
        self.lamda = lamda
        # Using the SGD optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.0)

    def fit(self, X, y):
        """
        Trains the SVM classifier with the given input and target labels.

        Params:
        X : The np array of (m, input_size) dimensions.
        y : The set of (m, ) target labels.
        """
        # Transform to tensors
        X = torch.Tensor(X) 
        y = torch.Tensor(y)

        # Creating dataset and dataloader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        for i in range(self.epochs):
            running_loss = 0
            # Iterating over each minibatch
            for data in dataloader:
                # Training pass
                inputs, labels = data
            
                self.optimizer.zero_grad()
                
                # Predicting the logit from the model
                output = self.model(inputs).squeeze()

                loss = self.criterion(output, labels)

                # Appending the L1 norm penalty to the loss
                loss += self.lamda * torch.norm(self.model.weight, 1)

                # This is where the model learns by backpropagating
                loss.backward()
                
                # And optimizes its weights here
                self.optimizer.step()
                
                running_loss += loss.item()
            else:
                continue
                print("Epoch {0}, Training loss: {1}".format(i+1, running_loss/len(dataloader)))
    
    def predict(self, X):
        """
        Predicts the labels for the given test inputs.

        Params:
        X : The set of (m, input_size) test points.

        Returns:
        predictions : The set of (m, ) predicted labels.
        """
        # Converting to tensor format
        X = torch.Tensor(X) 

        self.model.eval()

        # Turn off gradients for forward pass
        with torch.no_grad():
            logits = self.model(X)
            # Transforming logit to [-1,1] label and then rounding off to predict binary class
            predictions = logits.squeeze().detach().numpy()

            # SVM follows [+1, -1] labelling convention
            predictions[predictions >= 0] = 1
            predictions[predictions < 0] = -1

        return predictions
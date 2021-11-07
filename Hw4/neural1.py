import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
import time

# Transformations to normalize the PIL image
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Layer details for the neural network
input_size = 784
hidden_layer_size = 300
output_size = 10

losses = []
accuracies = []

class NeuralModel():
    """
    A class representing a simple 2-layer network, with SGD as an optimizer.
    """

    def __init__(self, sizes, epochs=20, l_rate=0.01):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = l_rate
        self.init_params()
        
    def sigmoid(self, x):
        """
        Performs the element-wise sigmoid activation on the given input array.

        Parameters:
        x : A numpy array

        Returns:

        The same numpy array with element-wise sigmoid applied to it.
        """
        return 1/(1 + np.exp(-x))
    
    def softmax(self, x):
        """
        Performs the element-wise softmax for each element in the input array x.

        Parameters:
        x : A numpy array 

        Returns :

        The same numpy array with softmax applied across the axis=1
        """
        exps = np.exp(x)
        denom = np.sum(exps, axis=1)
        denom.resize(exps.shape[0], 1)
        return exps / denom
        
    def init_params(self):
        """
        This function initializes the model parameters and creates the weight matrices.
        """
        input_layer = int(self.sizes[0])
        hidden_1 = int(self.sizes[1])
        output_layer = int(self.sizes[2])

        # Random initialization of weights between -1 and 1
        self.w1 = np.random.uniform(low=-1, high=1, size=(input_layer, hidden_1))
        self.w2 = np.random.uniform(low=-1, high=1, size=(hidden_1, output_layer))

        # Random initialization of weights from normal distribution
        #self.w1 = np.random.randn(input_layer, hidden_1)
        #self.w2 = np.random.randn(hidden_1, output_layer)

        # Zero initialization of weights
        #self.w1 = np.zeros((input_layer, hidden_1))
        #self.w2 = np.zeros((hidden_1, output_layer))
        
    def forward(self, inputs):
        """
        This function performs a simple foward pass through the model for the given minibatch.

        Parameters:
        inputs : A tensor of size=(batch_size, 784) containing the flattened inputs for each image in the dataset.

        Returns:
        self.out2 : A numpy array of size=(batch_size, 10) containing the softmax probabilities for the given minibatch.
        """
        # Input layer to hidden linear layer
        inputs = inputs.numpy()
        self.linear_1 = inputs.dot(self.w1)
        self.out1 = self.sigmoid(self.linear_1)

        # Hidden layer to softmax layer
        self.linear2 = self.out1.dot(self.w2)
        self.out2 = self.softmax(self.linear2)

        return self.out2
        
    def backward(self, x_train, y_train, output):
        """
        This function performs backpropagation by estimating the derivates at each layer and then applying chain rule to propagate it to the first layer.

        Parameters:
        x_train : The input tensor for the minibatch.
        y_train : The tensor containing the expected values for the minibatch.
        output : A numpy array containing the model's predictions for the minibatch

        Returns:
        w1_update : The delta matrix of the same size as W1 containing its derived updates.
        w2_update : The delta matrix of the same size as W2 containing its derived updates. 
        """
        # Convert tensors to numpy arrays
        x_train = x_train.numpy()
        y_train = y_train.numpy()

        batch_size = y_train.shape[0]

        # Derivative of loss 
        d_loss = output - y_train

        # Calculating delta for W2
        change_w2 = (1./batch_size) * np.matmul(self.out1.T, d_loss)

        # Backpropagating to the first layer from the second layer
        d_out_1 = np.matmul(d_loss, self.w2.T)
        d_linear_1 = d_out_1 * self.sigmoid(self.linear_1) * (1 - self.sigmoid(self.linear_1))

        # Calculating delta for W1
        change_w1 = (1. / batch_size) * np.matmul(x_train.T, d_linear_1)

        return change_w1, change_w2
    
    def update_weights(self, w1_update, w2_update):
        """
        This function takes the delta in the respective weight matrices and updates it according to the learning rate.

        Parameters:
        w1_update : The delta matrix of the same size as W1 containing its derived updates.
        w2_update : The delta matrix of the same size as W2 containing its derived updates. 
        """
        self.w1 -= self.l_rate * w1_update
        self.w2 -= self.l_rate * w2_update
    
    def compute_loss(self, y, y_hat):
        """
        This function computes the cross-entropy loss for a given minibatch of predictions.

        Parameters:
        y : The tensor containing the expected values for the minibatch.
        y_hat : The prediction of the model for the minibatch (in terms of softmax probability scores). 

        Returns:
        loss : The mean cross-entropy loss across the minibatch.
        """
        batch_size = y.shape[0]
        y = y.numpy()
        # Computing the cross entropy loss for the model and its given predictions
        loss = np.sum(np.multiply(y, np.log(y_hat)))
        loss = -(1./batch_size) * loss
        return loss
            
    def compute_metrics(self, val_loader):
        """
        Calculates the accuracy and mean loss of the model over the entire test set.

        Parameters:
        val_loader : The torch dataloader referencing the dataset's validation loader.

        Returns:
        (accuracy, loss) : A tuple representing the accuracy and the loss scalar representing the loss averages over all minibatches, respectively.
        """
        losses = []
        correct = 0
        total = 0

        for i, data in enumerate(val_loader):
            x, y = data
            # Converting to expected one-hot format
            y_onehot = torch.zeros(y.shape[0], 10)
            y_onehot[range(y_onehot.shape[0]), y]=1
            # Flattening input image into 1-D
            flattened_input = x.view(-1, 28*28)
            output = self.forward(flattened_input)
            predicted = np.argmax(output, axis=1)
            # Calculating correctly predicted labels
            correct += np.sum((predicted==y.numpy()))
            total += y.shape[0]
            # Computing the cross entropy loss
            loss = self.compute_loss(y_onehot, output)
            losses.append(loss)

        # Performing mean over all minibatches
        return (correct/total), np.mean(np.array(losses))
        
    def train(self, train_loader, val_loader):
        """
        This function trains the neural model over the epochs defined, updating its weights after each minibatch and also computing the test error after each epoch.

        Parameters:
        train_loader : The torch dataloader for the training set.
        val_loader : The torch dataloader for the test set.
        """
        start_time = time.time()
        global losses
        global accuracies
        for iteration in range(self.epochs):
            for i, data in enumerate(train_loader):
                x, y = data
                # Since the model is producing a softmax probability over 10 classes, the label needs to be converted to a one-hot encoded vector
                y_onehot = torch.zeros(y.shape[0], 10)
                y_onehot[range(y_onehot.shape[0]), y]=1
                # Converting 28x28 image into a flattened input
                flattened_input = x.view(-1, 28*28)
                # Forward pass the input through the model
                output = self.forward(flattened_input)
                # Compute gradients for the linear layer weights using SGD
                w1_update, w2_update = self.backward(flattened_input, y_onehot, output)
                # Perform weight update for the minibatch
                self.update_weights(w1_update, w2_update)
            # Compute the mean loss over the test set after the completion of epoch
            accuracy, loss = self.compute_metrics(val_loader)
            losses.append(loss)
            accuracies.append(accuracy)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%, Loss: {3:.2f}'.format(
                iteration+1, time.time() - start_time, accuracy*100, loss
            ))
    
if __name__=='__main__':
    model = NeuralModel(sizes=[784, 300, 10], epochs=20)
    # Download and load the training data using PyTorch's dataloader
    trainset = datasets.MNIST('./dataset/MNIST/', download=True, train=True, transform=transform)
    valset = datasets.MNIST('./dataset/MNIST/', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=True)
    # Training the model over the MNIST dataset
    model.train(train_loader=trainloader, val_loader=valloader)
    plt.xlabel('Epochs')
    plt.ylabel('Test Loss')
    plt.plot(losses)
    plt.show()

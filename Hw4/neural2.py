import numpy as np
import time
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn
import torch

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('./dataset/MNIST/', download=True, train=True, transform=transform)
valset = datasets.MNIST('./dataset/MNIST/', download=True, train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=True)

# Layer details for the neural network
input_size = 784
hidden_size = 300
output_size = 10

# Build a simple 2-layer feed forward network as described
model = nn.Sequential(nn.Linear(input_size, hidden_size, bias=False),
                      nn.Sigmoid(),
                      nn.Linear(hidden_size, output_size, bias=False),
                      nn.LogSoftmax(dim=1))
print(model)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
# Using the cross entropy (or NLL) loss
criterion = nn.NLLLoss()

epochs = 20
losses = []
for i in range(epochs):
    running_loss = 0
    for images, labels in train_loader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
    else:
        losses.append(float(running_loss/len(train_loader)))
        print("Epoch {0}, Training loss: {1}".format(i, running_loss/len(train_loader)))

correct_count, all_count = 0, 0
for images,labels in val_loader:
  for i in range(len(labels)):
    img = images[i].view(1, 28*28)
    # Turn off gradients for forward pass
    with torch.no_grad():
        logps = model(img)

    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))
plt.xlabel('Epochs')
plt.ylabel('Test Loss')
plt.plot(losses)
plt.show()

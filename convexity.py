#PROTOTYPE - Understanding Convexity of Representations
# This is what i understood from thhe task you assigned me
# I used RELU and optimiser which according to the paper helps the convexity (i don't know if there's away without using it)


#import Libraries
import os
import torch
print(torch.__version__)
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np


# Get Device 
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device")

# Get Fashion MNIST DATASET
#for training
fashionMNIST_trainset = datasets.FashionMNIST(root = './data/', train=True, download=True, transform=None)

#for testing
fashionMNIST_testset = datasets.FashionMNIST(root='./data/', train=False, download=True, transform=None)

#DATA Expolration
print(len(fashionMNIST_trainset)) #60000
print(len(fashionMNIST_testset)) #10000

#Data visualization
image, label = fashionMNIST_trainset[0]  # Get the first image and its label.
plt.imshow(image, cmap='gray')    # Plot the image.
plt.title(f'Label: {label}')      # Set the title as the label of the image.
plt.show()                        # Display the plot.


#Build a Neural Network
class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MyNeuralNetwork().to(device)



# Apply the Model

# Define a transform to normalize the data
transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,)) ])

# Apply the transforms on the dataset
fashionMNIST_trainset = datasets.FashionMNIST(root='./data/', train=True, download=True, transform=transform)
fashionMNIST_testset = datasets.FashionMNIST(root='./data/', train=False, download=True, transform=transform)

# DataLoader
train_loader = torch.utils.data.DataLoader(fashionMNIST_trainset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(fashionMNIST_testset, batch_size=64, shuffle=False)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for images, labels in train_loader:
        # Move data to the appropriate device
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
extracted_features = []
extracted_labels = []
predicted_classes = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        extracted_features.append(outputs.cpu())
        extracted_labels.append(labels.cpu())
        predicted_classes.append(predicted.cpu())

print(f'Accuracy of the model on the 10000 test images: {100 * correct / total} %')

# Convert list of tensors to a single tensor
extracted_features = torch.cat(extracted_features).numpy()
extracted_labels = torch.cat(extracted_labels).numpy()
predicted_classes = torch.cat(predicted_classes).numpy()

# Initialize and fit the NearestNeighbors model
neighbors_model = NearestNeighbors(n_neighbors=6)
neighbors_model.fit(extracted_features)

# Find the nearest neighbors for each point
distances, indices = neighbors_model.kneighbors(extracted_features)

convexity = []
# Analyze convexity
for i, location in enumerate(indices):
    original_class = extracted_labels[i]
    predicted_class = predicted_classes[i]  # This is the actual predicted class
    neighbor_classes = extracted_labels[location]
    same_class_count = np.sum(neighbor_classes == original_class)
    is_convex = same_class_count > len(neighbor_classes) / 2 # This calculates half of the number of neighbors to check if a point is in a region where its own class is the majority
    print(f"Embedding {i}: Original class: {original_class}, Actual Class: {predicted_class}, Convexity: {'Yes' if is_convex else 'No'}")


# Dimensionality reduction with t-SNE
tsne = TSNE(n_components=2, random_state=0)
features_2d = tsne.fit_transform(extracted_features)

# Plotting t-SNE
plt.figure(figsize=(10, 10))
for i in range(10):  # Assuming 10 classes
    indices = extracted_labels == i
    plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=f'Class {i}')
plt.legend()
plt.show()









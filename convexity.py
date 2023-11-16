#PROTOTYPE - Understanding Convexity of Representations
# This is what i understood from thhe task you assigned me
#I used RELU and optimiser which according to the paper helps the convexity (i don't know if there's away without using it)


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
import pandas as pd
import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform


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

    def forward(self, x, return_embedding=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 14 * 14)
        embedding = F.relu(self.fc1(x))
        if return_embedding:
            return embedding
        x = self.fc2(embedding)
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for images, labels in train_loader:
        # Move data to the appropriate device
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        embeddings = model(images, return_embedding=True)
        loss = criterion(embeddings, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model
model.eval()  # Set the model to evaluation mode

# Extract features and labels
extracted_embeddings = []
extracted_labels = []

with torch.no_grad(): 
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        embeddings = model(images, return_embedding=True)
        extracted_embeddings.append(embeddings.cpu())
        extracted_labels.append(labels.cpu())

        _, predicted = torch.max(embeddings.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

extracted_embeddings = torch.cat(extracted_embeddings).numpy()
extracted_labels = torch.cat(extracted_labels).numpy()

# Dimensionality reduction with t-SNE
tsne = TSNE(n_components=2, random_state=0)
features_2d = tsne.fit_transform(extracted_embeddings)


print(f'Accuracy of the model on the 10000 test images: {100 * correct / total} %')

# Plotting
plt.figure(figsize=(10, 10))
for i in range(10):  # Assuming 10 classes
    indices = extracted_labels == i
    plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=f'Class {i}', alpha=0.5)
plt.legend()
plt.title('t-SNE visualization of image features')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()

# Applying Euclidean Convexity
# Function to calculate Euclidean distance efficiently
def euclidean_distance_matrix(data):
    return squareform(pdist(data, 'euclidean'))

# Select a random batch of embeddings
batch_size = 1000  # Adjust this based on your requirements
if len(features_2d) > batch_size:
    indices = np.random.choice(len(features_2d), batch_size, replace=False)
    selected_features = features_2d[indices]
    selected_labels = extracted_labels[indices]
else:
    selected_features = features_2d
    selected_labels = extracted_labels

# Create a graph
G = nx.Graph()

# Efficient calculation of distance matrix for the selected batch
dist_matrix = euclidean_distance_matrix(selected_features)
threshold = 6

# Add nodes and edges based on Euclidean convexity for the selected batch
for i in range(len(selected_features)):
    for j in range(i+1, len(selected_features)):
        if dist_matrix[i, j] < threshold:
            G.add_edge(i, j)

# Visualization
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G)  # Calculate positions for a better layout
nx.draw(G, pos, node_color=[selected_labels[n] for n in G.nodes], with_labels=False, node_size=30, cmap=plt.cm.Set1)
plt.title('Batched NetworkX Graph Visualization of Euclidean Convexity in Embeddings')
plt.show()
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
import random


# Get Device 
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device")


# Seed value
seed_value = 200

# Set the seed for PyTorch
torch.manual_seed(seed_value)

# Set the seed for NumPy
np.random.seed(seed_value)

# Set the seed for Python's random module (if used)
random.seed(seed_value)


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



# Applying the Model

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

# Training loop
num_epochs = 20
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


#Testing the embeddings on a KNN classifier
'''
By design, kNN classifiers divide the input space into convex 
regions, where each region corresponds to a class. Since kNN 
classifiers inherently create convex regions, the expectation is 
that when measuring convexity using this algorithm, it should show
maximal convexity for a kNN classifier. This expectation would serve 
as a benchmark.
'''

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

extracted_embeddings_2 = extracted_embeddings
extracted_labels_2 = extracted_labels

# Split the embeddings and labels into training and test sets
X_train, X_test, y_train, y_test = train_test_split(extracted_embeddings_2, extracted_labels_2, test_size=0.2, random_state=seed_value)

knn_alpha = KNeighborsClassifier(n_neighbors=6)
# Train the kNN classifier
knn_alpha.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_alpha.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Optional: Display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Apply t-SNE to reduce dimensions to 2D for visualization
tsne_model = TSNE(n_components=2, random_state=42)
X_test_2d = tsne_model.fit_transform(X_test)

# Plot the results
plt.figure(figsize=(10, 10))
for i in range(10):  # Assuming 10 classes
    # Find the indices of predictions for each class
    indices = y_pred == i
    # Scatter plot for each class with a different color and label
    plt.scatter(X_test_2d[indices, 0], X_test_2d[indices, 1], label=f'Class {i}', alpha=0.5)

# Add legend, title, and labels
plt.legend()
plt.title('t-SNE Visualization of kNN Predicted Classes on Test Set')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()


#Measuring Euclidean Convexity + knn classifier
knn_beta = KNeighborsClassifier(n_neighbors=6)
knn_beta.fit(extracted_embeddings_2, extracted_labels_2)


def measure_convexity(extracted_embeddings, extracted_labels, knn, device, num_pairs=500, num_interpolated_points=6):
    """
    Measures the convexity of representations for each class in the dataset.

    Args:
    extracted_embeddings (numpy.ndarray): Embeddings extracted from the model.
    extracted_labels (numpy.ndarray): Corresponding labels of the embeddings.
    model (torch.nn.Module): Trained neural network model.
    device (torch.device): Device to run the model (CPU or CUDA).
    num_pairs (int): Number of pairs to sample per class.
    num_interpolated_points (int): Number of interpolated points between each pair.
    """
    
    class_labels = range(10)  # Assuming 10 classes

    for class_label in class_labels:
        class_counts = 0
        class_embeddings = extracted_embeddings_2[extracted_labels_2 == class_label]

        for _ in range(num_pairs):
            # Randomly select two different embeddings from the class
            idxs = np.random.choice(len(class_embeddings), 2, replace=False)
            z1, z2 = class_embeddings[idxs[0]], class_embeddings[idxs[1]]

            for i in range(num_interpolated_points + 1):
                λ = i / num_interpolated_points
                z_prime = λ * z1 + (1 - λ) * z2

                 # Predict with kNN classifier
                predicted_class = knn_beta.predict([z_prime])[0]

                # Check if the predicted class is the same as class y
                if predicted_class == class_label:
                    class_counts += 1

        # Calculate the proportion for each class
        proportion = class_counts / ((num_interpolated_points + 1) * num_pairs)
        print("Proportion of points predicted as class {}: {}".format(class_label, proportion))


measure_convexity(extracted_embeddings_2, extracted_labels_2, knn_beta, device)

#Measuring convexity without KNN
def measure_convexity(extracted_embeddings, extracted_labels, model, device, num_pairs=500, num_interpolated_points=6):
    """
    Measures the convexity of representations for each class in the dataset.

    Args:
    extracted_embeddings (numpy.ndarray): Embeddings extracted from the model.
    extracted_labels (numpy.ndarray): Corresponding labels of the embeddings.
    model (torch.nn.Module): Trained neural network model.
    device (torch.device): Device to run the model (CPU or CUDA).
    num_pairs (int): Number of pairs to sample per class.
    num_interpolated_points (int): Number of interpolated points between each pair.
    """
    model.eval()  # Ensuring the model is in evaluation mode

    class_labels = range(10)  # Assuming 10 classes

    for class_label in class_labels:
        class_counts = 0
        class_embeddings = extracted_embeddings[extracted_labels == class_label]

        for _ in range(num_pairs):
            # Randomly select two different embeddings from the class
            idxs = np.random.choice(len(class_embeddings), 2, replace=False)
            z1, z2 = class_embeddings[idxs[0]], class_embeddings[idxs[1]]

            for i in range(num_interpolated_points + 1):
                λ = i / num_interpolated_points
                z_prime = λ * z1 + (1 - λ) * z2

                # Convert to tensor and predict
                z_prime_tensor = torch.from_numpy(z_prime).float().unsqueeze(0).to(device)
                predictions = model.fc2(z_prime_tensor)
                predicted_class = torch.argmax(predictions, dim=1)

                # Check if the predicted class is the same as class y
                if predicted_class.item() == class_label:
                    class_counts += 1

        # Calculate the proportion for each class
        proportion = class_counts / ((num_interpolated_points + 1) * num_pairs)
        print("Proportion of points predicted as class {}: {}".format(class_label, proportion))

measure_convexity(extracted_embeddings, extracted_labels, model, device)
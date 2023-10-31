import os
import math
import collections
import csv
import re

######################################################
# Section 3: Language Identification with Naive Bias #
######################################################

##### part 1 #####

dataset_path = "C:/Users/Ariana Negreiro/Dropbox/Classes/CS760/hw/hw4/languageID/"

# Step 1: Load and preprocess the dataset
def load_dataset(dataset_path):
    data = {}

    for i in os.listdir(dataset_path):
        lang = i[0]

        with open(os.path.join(dataset_path, i), 'r', encoding='utf-8') as file:
            text = file.read().lower()  # Convert to lowercase
            # Remove control characters such as new-line
            #text = text.replace("\n", "")
            text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)
            if lang not in data:
                data[lang] = []
            data[lang].append(text)

        #text = i.read().decode("utf-8").lower()

    return data


# Step 2: Prepare the training data
def prepare_training_data(data):
    training_data = {lang: [] for lang in data.keys()}
    for lang, documents in data.items():
        for doc in documents[:10]:  # Use files 0.txt to 9.txt for training
            training_data[lang].append(doc)
    return training_data

# Step 3: Estimate prior probabilities with additive smoothing
def estimate_prior_probabilities(data, smoothing_parameter=0.5):
    prior_probabilities = {}
    total_documents = sum(len(docs) for docs in data.values())
    for lang, documents in data.items():
        smoothed_count = len(documents) + smoothing_parameter
        prior_probabilities[lang] = math.log(smoothed_count / (total_documents + smoothing_parameter * len(data)))
    return prior_probabilities

# Step 4: Implement the Naive Bayes classification
def classify_document(document, prior_probs):
    char_counts = collections.Counter(document)
    predicted_language = None
    max_log_prob = float('-inf')

    for lang, prior_prob in prior_probs.items():
        log_prob = prior_prob
        for char, count in char_counts.items():
            # Use Laplace (additive) smoothing
            prob = (count + 0.5) / (len(training_data[lang]) + 0.5 * 27)
            log_prob += count * math.log(prob)
        if log_prob > max_log_prob:
            max_log_prob = log_prob
            predicted_language = lang
    return predicted_language

# Step 5: Load the dataset and perform classification
data = load_dataset(dataset_path)
training_data = prepare_training_data(data)
prior_probs = estimate_prior_probabilities(training_data)

# Print and include prior probabilities in your final report
for lang, prob in prior_probs.items():
    print(f'Estimated prior probability for {lang}: {math.exp(prob)}')

# Now you can classify new documents using the classify_document function
# Example: predicted_lang = classify_document(new_document, prior_probs)

##### part 2 #####

def estimate_class_conditional_probabilities(lang_data, smoothing_parameter=0.5):
    char_counts = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}
    total_chars = 0

    for doc in lang_data:
        char_counts.update(collections.Counter(doc))
        total_chars += len(doc)

    theta_e = []
    for char in 'abcdefghijklmnopqrstuvwxyz ':
        smoothed_count = char_counts[char] + smoothing_parameter
        prob = smoothed_count / (total_chars + smoothing_parameter * 27)
        theta_e.append(prob)

    return theta_e

# Save theta_e as a CSV file
def save_theta_as_csv(theta_e, filename):
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(list('abcdefghijklmnopqrstuvwxyz'))
        writer.writerow(theta_e)

# Estimate class conditional probabilities for each language
english_training_data = training_data['e']  # English training data
japanese_training_data = training_data['j']  # Japanese training data
spanish_training_data = training_data['s']  # Spanish training data

theta_e = estimate_class_conditional_probabilities(english_training_data)
theta_j = estimate_class_conditional_probabilities(japanese_training_data)
theta_s = estimate_class_conditional_probabilities(spanish_training_data)

# Print thetas
print("Estimated class conditional probabilities for Japanese (theta_j):", theta_j)
print("Estimated class conditional probabilities for Spanish (theta_s):", theta_s)
print("Estimated class conditional probabilities for English (theta_e):", theta_e)

# Save theta_e as a CSV file
save_theta_as_csv(theta_e, 'theta_e.csv')
save_theta_as_csv(theta_j, 'theta_j.csv')
save_theta_as_csv(theta_s, 'theta_s.csv')



##### part 3 #####

# Step 1: Read and preprocess the test document (e10.txt)
with open(os.path.join(dataset_path, 'e10.txt'), 'r', encoding='utf-8') as file:
    test_document = file.read().lower()  # Convert to lowercase
    #test_document = test_document.replace("\n", "")
    test_document = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", test_document)

# Step 2: Create a bag-of-words count vector
# find all unique words
#words = test_document.split()
#word_count = {}
#for word in words:
#    word = ''.join(e for e in word if e.isalnum())
#    if word:
#        if word in word_count:
#            word_count[word] += 1
#        else:
#            word_count[word] = 1

# Sort the dictionary by counts in descending order
#sorted_word_count = {k: v for k, v in sorted(word_count.items(), key=lambda item: item[1], reverse=True)}

# Create a list of the 27 words with the highest counts
#top_words = list(sorted_word_count.keys())[:27]

# Print the top words and their counts
#for word in top_words:
#    print(f"{word}: {sorted_word_count[word]}")

# Step 2: Create a bag-of-words (bag-of-letters?) count vector 
def create_bow(text):
    letter_count = {}
    tokens = list(text)
    for letter in tokens:
        if letter in letter_count:
            letter_count[letter] +=1
        else:
            letter_count[letter] = 1
    sorted_letters = {k: v for k, v in sorted(letter_count.items(), key=lambda item: item[1], reverse=True)}

    return sorted_letters


sorted_letters = create_bow(test_document)

# Print the letters and their counts
for letter in sorted_letters:
    print(f"{letter}: {sorted_letters[letter]}")



##### part 5 #####
#x = sorted_letters # defined as x in previous problem

#sorted_dict = dict(sorted(x.items(), key=lambda item: (item[0] == ' ', item[0])))

#x = list(sorted_dict.values())

x = estimate_class_conditional_probabilities(test_document)

# thetas
def calculate_language_probability(x, theta):
    probability = 1.0
    for i in range(len(x)):
        probability *= (theta[i] ** x[i])
    return probability

#def calculate_language_probability(x, theta):
#    log_probability = 0.0
#    for i in range(len(x)):
#        log_probability += x[i] * math.log(theta[i])
#    return math.exp(log_probability)

english_probability = calculate_language_probability(x, theta_e)
japanese_probability = calculate_language_probability(x, theta_j)
spanish_probability = calculate_language_probability(x, theta_s)

# Print the probabilities for each language
print(f"Estimated probability for English (y=e): {english_probability:.{5}f}")
print(f"Estimated probability for Japanese (y=e): {japanese_probability:.{5}f}")
print(f"Estimated probability for Spanish (y=e): {spanish_probability:.{5}f}")


##### part 6 #####

# prior probs
prior_probs = {'e': 0.33, 'j': 0.33, 's': 0.33}  # Replace with your estimated prior probabilities

# Calculate the posterior probabilities for each class using Bayes' rule
posterior_probs = {}
for lang in prior_probs.keys():
    likelihood = calculate_language_probability(x, theta_e if lang == 'e' else theta_j if lang == 'j' else theta_s)
    posterior_probs[lang] = likelihood * prior_probs[lang]

# Normalize the posterior probabilities
total_posterior = sum(posterior_probs.values())
posterior_probs = {lang: prob / total_posterior for lang, prob in posterior_probs.items()}

# Predict the class label
predicted_class = max(posterior_probs, key=posterior_probs.get)

# Print the posterior probabilities and predicted class label
print("Posterior probabilities:")
for lang, prob in posterior_probs.items():
    print(f"Estimated probability for {lang} (y={lang} | x): {prob:.4f}")
print("Predicted class label for x:", predicted_class)


##### part 7 #####

test_dataset_path = "C:/Users/Ariana Negreiro/Dropbox/Classes/CS760/hw/hw4/languageID/"

# Define class labels for English, Japanese, and Spanish
class_labels = ['e', 'j', 's']

# Initialize the confusion matrix
confusion_matrix = {true_lang: {pred_lang: 0 for pred_lang in class_labels} for true_lang in class_labels}

# Step 1: Load and preprocess the test set
def load_test_set(dataset_path):
    data = {}

    for lang in class_labels:
        data[lang] = []
        for i in range(10, 20):
            file_name = f"{lang.lower()}{i}.txt"
            #print(file_name)
            with open(os.path.join(dataset_path, file_name), 'r', encoding='utf-8') as file:
                text = file.read().lower()
                text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)
                data[lang].append(text)

    return data

test_data = load_test_set(test_dataset_path)

# Create a dictionary to store the probabilities
class_probabilities = {}


# Step 2: Classify each document and update the confusion matrix
for true_lang in class_labels:
    #class_probabilities[true_lang] = calculate_language_probability(x, locals()[f'theta_{lang}'])
    for document in test_data[true_lang]:
        #print(list(document))
        # Classify the document using your classifier
        x = estimate_class_conditional_probabilities(list(document))
        #print(x)
        #sorted_dict = dict(sorted(x.items(), key=lambda item: (item[0] == ' ', item[0])))
        #x = list(sorted_dict.values())
        #print(x)
        e_probability = calculate_language_probability(x, theta_e)
        j_probability = calculate_language_probability(x, theta_j)
        s_probability = calculate_language_probability(x, theta_s)

        # Store the class probabilities in the dictionary
        class_probabilities['e'] = e_probability
        class_probabilities['j'] = j_probability
        class_probabilities['s'] = s_probability

        # Determine the predicted class label based on the maximum probability
        predicted_class = max(class_labels, key=class_probabilities.get)

        # Determine the predicted class label
        #predicted_class = max(class_labels, key=lambda lang: locals()[f'{lang.lower()}_probability'])

        # Update the confusion matrix
        confusion_matrix[true_lang][predicted_class] += 1

# Print the confusion matrix
print("Confusion Matrix:")
for true_lang in class_labels:
    row = [confusion_matrix[true_lang][pred_lang] for pred_lang in class_labels]
    print(f"{true_lang}:", row)


######################################################
# Section 4: Simple Feed-Forward Network #
######################################################

##### part 1 #####

import numpy as np
import torch
from torchvision import datasets, transforms
import csv

# Define hyperparameters
learning_rate = 0.01
num_epochs = 10
batch_size = 64

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Define the neural network architecture manually
input_size = 28 * 28
hidden_size = 128
output_size = 10

# Initialize weights and biases
W1 = torch.randn(input_size, hidden_size, requires_grad=True)
b1 = torch.zeros(1, hidden_size, requires_grad=True)
W2 = torch.randn(hidden_size, output_size, requires_grad=True)
b2 = torch.zeros(1, output_size, requires_grad=True)

#part 3, initialize all weights to 0 or randomly between -1 and 1
W1 = torch.zeros_like(W1)
W2 = torch.zeros_like(W2)

W1 = (torch.rand(input_size, hidden_size) * 2 - 1).requires_grad_()
W2 = (torch.rand(hidden_size, output_size) * 2 - 1).requires_grad_()

# Create a list to store the learning curve values
learning_curve = []
test_errors = []

# Training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, input_size)

        # Forward pass
        z1 = torch.matmul(images, W1) + b1
        a1 = 1 / (1 + torch.exp(-z1))
        z2 = torch.matmul(a1, W2) + b2
        a2 = torch.exp(z2) / torch.exp(z2).sum(dim=1, keepdim=True)

        # Compute the loss
        one_hot_labels = torch.zeros(len(labels), output_size)
        one_hot_labels[range(len(labels)), labels] = 1
        loss = -torch.sum(one_hot_labels * torch.log(a2)) / batch_size

        # Backward pass
        grad_a2 = (a2 - one_hot_labels) / batch_size
        grad_W2 = torch.matmul(a1.t(), grad_a2)
        grad_b2 = torch.sum(grad_a2, dim=0, keepdim=True)
        grad_a1 = torch.matmul(grad_a2, W2.t())
        grad_z1 = grad_a1 * a1 * (1 - a1)
        grad_W1 = torch.matmul(images.t(), grad_z1)
        grad_b1 = torch.sum(grad_z1, dim=0, keepdim=True)

        # Update weights and biases using gradient descent without in-place operations
        W1 = W1 - learning_rate * grad_W1
        b1 = b1 - learning_rate * grad_b1
        W2 = W2 - learning_rate * grad_W2
        b2 = b2 - learning_rate * grad_b2

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}')
            
        # Append the current loss to the learning curve
        learning_curve.append([epoch + (i + 1) / len(train_loader), loss.item()])



# Evaluate the model
# Define the test DataLoader
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

correct = 0
total = 0

for images, labels in test_loader:
    images = images.view(-1, input_size)
    z1 = torch.matmul(images, W1) + b1
    a1 = 1 / (1 + torch.exp(-z1))
    z2 = torch.matmul(a1, W2) + b2
    a2 = torch.exp(z2) / torch.exp(z2).sum(dim=1, keepdim=True)

    _, predicted = torch.max(a2.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print(f'Accuracy on the test set: {100 * correct / total}%')

accuracy = 100 * correct / total
test_loss = -torch.sum(torch.log(a2[range(len(labels)), labels])) / total
print(f'Test Loss: {test_loss.item()}')

# Append the test error to the list
test_errors.append([num_epochs, test_loss.item(), accuracy])

# Save the learning curve and test errors to a CSV file
with open('learning_curve_and_test_errors_weights_0.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Loss', 'Test Loss', 'Test Accuracy'])
    writer.writerows(learning_curve)
    writer.writerows(test_errors)


##### part 2 #####

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define hyperparameters
learning_rate = 0.01
num_epochs = 10
batch_size = 64

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the neural network using PyTorch's nn.Module
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=1)

# Initialize the neural network
input_size = 28 * 28
hidden_size = 128
output_size = 10
model = NeuralNetwork(input_size, hidden_size, output_size)

# Define the loss function (cross-entropy)
criterion = nn.CrossEntropyLoss()

# Define the optimizer (Adam)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create lists to store the learning curve and test errors
learning_curve = []
test_errors = []

# Training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, input_size)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}')
            
        # Append the current loss to the learning curve
        learning_curve.append([epoch + (i + 1) / len(train_loader), loss.item()])


# Evaluate the model
model.eval()
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

correct = 0
total = 0
test_loss = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(-1, input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        test_loss += criterion(outputs, labels).item()

    accuracy = 100 * correct / total
    test_loss /= len(test_loader)  # Average test loss

    print(f'Accuracy on the test set: {accuracy}%')
    print(f'Average Test Loss: {test_loss}')

# Append the test error to the list
test_errors.append([num_epochs, test_loss, accuracy])

# Save the learning curve and test errors to a CSV file
with open('learning_curve_and_test_errors.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Loss', 'Test Loss', 'Test Accuracy'])
    writer.writerows(learning_curve)
    writer.writerows(test_errors)
    
    
##### plot graphs #####

import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv(r'C:\Users\arneg\Dropbox\Classes\CS760\hw\hw4\learning_curve_and_test_errors_weights_1.csv')

# Extract the x-axis data (assuming the first column is x-axis)
x = data.iloc[:, 0]

# Create line plots for each column (y-axis data)
for col in data.columns[1:]:
    y = data[col]
    plt.plot(x, y, label=col)

# Add labels and legend
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Show the plot
plt.show()

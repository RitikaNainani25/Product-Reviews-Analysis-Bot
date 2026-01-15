import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# Load intents
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # Add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        w = tokenize(pattern)
        # Add to our words list
        all_words.extend(w)
        # Add to xy pair
        xy.append((w, tag))

# Stem and lower each word
ignore_words = ['?', '.', '!', ',', "'", '"']
all_words = [stem(w.lower()) for w in all_words if w not in ignore_words]
# Remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(f"Total patterns: {len(xy)}")
print(f"Total tags: {len(tags)}")
print(f"Tags: {tags}")
print(f"Unique stemmed words: {len(all_words)}")

# Create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int64)

# Hyper-parameters - Adjusted for larger dataset
num_epochs = 2000  # Increased for better training
batch_size = 16     # Increased batch size
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 16    # Increased hidden size
output_size = len(tags)

print(f"\nModel Architecture:")
print(f"Input size: {input_size}")
print(f"Hidden size: {hidden_size}")
print(f"Output size: {output_size}")
print(f"Training samples: {len(X_train)}")

class ChatDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.n_samples = len(X_data)
        self.x_data = torch.from_numpy(X_data)
        self.y_data = torch.from_numpy(y_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Create dataset and data loader
dataset = ChatDataset(X_train, y_train)
train_loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2 if torch.cuda.is_available() else 0
)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)  # Learning rate scheduler

# Training loop
print("\nStarting training...")
for epoch in range(num_epochs):
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Update learning rate
    scheduler.step()
    
    # Calculate epoch statistics
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct_predictions / total_predictions
    
    # Print progress
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1:04d}/{num_epochs}], '
              f'Loss: {avg_loss:.6f}, '
              f'Accuracy: {accuracy:.2f}%, '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')

# Final evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        outputs = model(words)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'\nFinal Training Accuracy: {100 * correct / total:.2f}%')

# Prepare data to save
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags,
    "intents": intents  # Save the entire intents structure for reference
}

# Save the model
FILE = "data.pth"
torch.save(data, FILE)
print(f'\nTraining complete! Model saved to {FILE}')

# Additional info
print(f"\nModel Summary:")
print(f"- Input layer: {input_size} neurons")
print(f"- Hidden layer: {hidden_size} neurons")
print(f"- Output layer: {output_size} neurons (one for each tag)")
print(f"- Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"- Tags trained: {tags}")

# Test with some sample inputs
print("\nSample predictions (for testing):")
test_sentences = [
    "hello there",
    "thanks for your help",
    "this product is amazing",
    "i need help",
    "how much does it cost?",
    "tell me a joke"
]

for sentence in test_sentences:
    sentence_tokens = tokenize(sentence)
    X = bag_of_words(sentence_tokens, all_words)
    X = torch.from_numpy(X.reshape(1, X.shape[0])).float().to(device)
    
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    # Get a random response for this tag
    for intent in intents['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            break
    
    print(f"Input: '{sentence}' -> Tag: {tag} -> Response: {response}")
import torch
import torch.nn as nn
import torch.optim as optim

#  Sample data: [sentence, label]
# Label: 1 = Positive, 0 = Negative
data = [
    ("I love this product", 1),
    ("This is amazing", 1),
    ("Very happy with the service", 1),
    ("I hate this", 0),
    ("This is terrible", 0),
    ("Not good at all", 0),
]

#  Preprocessing - simple tokenizer
def tokenize(sentence):
    return sentence.lower().split()

# Build vocabulary
vocab = set()
for sentence, _ in data:
    vocab.update(tokenize(sentence))

word2idx = {word: idx for idx, word in enumerate(vocab)}
vocab_size = len(word2idx)

def vectorize(sentence):
    vec = torch.zeros(vocab_size)
    for word in tokenize(sentence):
        if word in word2idx:
            vec[word2idx[word]] += 1
    return vec

# Prepare dataset
X = torch.stack([vectorize(s) for s, _ in data])
y = torch.tensor([label for _, label in data], dtype=torch.long)

#  Define simple model
class SentimentModel(nn.Module):
    def __init__(self, input_size):
        super(SentimentModel, self).__init__()
        self.fc = nn.Linear(input_size, 2)  # Output: 2 classes (positive/negative)

    def forward(self, x):
        return self.fc(x)

model = SentimentModel(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

#  Train the model
epochs = 20
for epoch in range(epochs):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

print(" Training complete!")

# Test the model on custom sentences
def predict(sentence):
    with torch.no_grad():
        vec = vectorize(sentence)
        output = model(vec)
        predicted = torch.argmax(output).item()
        label = "Positive" if predicted == 1 else "Negative"
        print(f"Sentence: \"{sentence}\" → Prediction: {label}")

# Test predictions
print("\n--- Testing the model ---")
predict("I love it")
predict("Worst service ever")
predict("Not bad")
predict("Absolutely fantastic")
predict("I will never buy this again")

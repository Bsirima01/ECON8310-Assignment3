import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

# Define custom dataset class
class CustomFashionMNIST(Dataset):
    def __init__(self, train=True, transform=None):
        self.data = datasets.FashionMNIST(root='./data', train=train, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalization based on Fashion MNIST stats
])

# Create custom data loaders
train_loader = DataLoader(CustomFashionMNIST(train=True, transform=transform), batch_size=64, shuffle=True)
test_loader = DataLoader(CustomFashionMNIST(train=False, transform=transform), batch_size=64, shuffle=False)
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class FashionMNISTNet(nn.Module):
    def __init__(self):
        super(FashionMNISTNet, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten image
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.output(x)
        return x

# Initialize model, loss function, and optimizer
model = FashionMNISTNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (basic setup)
num_epochs = 5
for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
torch.save(model.state_dict(), 'fashion_mnist_model.pth')
# Load Model Script
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define model architecture as before (reuse FashionMNISTNet class)

# Initialize the model and load weights
model = FashionMNISTNet()
model.load_state_dict(torch.load('fashion_mnist_model.pth'))
model.eval()

# Define test data loader
test_loader = DataLoader(CustomFashionMNIST(train=False, transform=transform), batch_size=64, shuffle=False)

# Evaluation loop
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the model on the test images: {100 * correct / total:.2f}%")


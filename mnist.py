# -*- coding: utf-8 -*-

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time
import psutil
import os
# Function to calculate memory usage
# Function to calculate memory usage
def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

# Start recording time and memory
start_time = time.time()
start_memory = memory_usage()

# First, define the defense layer
class DefenseLayer(nn.Module):
    def __init__(self, beta=0.1, gamma=1.0):  # reduce beta to lower the noise level
        super().__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, x):
        # Compute reverse sigmoid perturbation
        z = self.gamma * torch.sigmoid(x).log()
        r = self.beta * (torch.sigmoid(z) - 0.5)
        y_tilde = x-r

        # Normalize y_tilde to sum to 1
        y_tilde = y_tilde / y_tilde.sum(dim=1, keepdim=True)
        return y_tilde

# Modify Lenet class to include the defense layer
class Lenet(nn.Module):
    def __init__(self, beta=0.1, gamma=1.0):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.defense = DefenseLayer(beta, gamma)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        output = self.defense(F.softmax(x, dim=1))
        output = torch.log(output + 1e-10) # Add a small value to prevent log(0) issue
        return output


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load dataset
dataset1 = datasets.MNIST('./data', train=True, download=True,
                          transform=transform)
dataset2 = datasets.MNIST('./data', train=False,
                          transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=64, shuffle=True)
# Build the model we defined above
model = Lenet()

#Define the optimizer for model training
optimizer = optim.Adadelta(model.parameters(), lr=1)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

loss_fn = nn.NLLLoss()  # Define loss function

model.train()
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)  # Use loss function
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    scheduler.step()

model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        test_loss += loss_fn(output, target).item() * data.size(0)  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

torch.save(model.state_dict(), "mnist_cnn.pt")
# End recording time and memory
end_time = time.time()
end_memory = memory_usage()
# Calculate and print time and memory usage
print('Time taken: ', end_time - start_time, 'seconds')
print('Memory used: ', (end_memory - start_memory) / (1024 * 1024), 'MB')
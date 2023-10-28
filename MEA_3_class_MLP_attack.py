# -*- coding: utf-8 -*-

from __future__ import print_function

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
import random
import copy
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

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

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
        return F.log_softmax(x, dim=1)

# Define two model
model = LeNet()
extracted_model = LeNet()

# Define normalization
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

# Load MNIST dataset
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# Load target model
model.load_state_dict(torch.load("mnist_cnn.pt"))

# querry the target model and obtain the response
model.eval()
attack_number = 200 # maximum attack_number is 120

attack_querries_mask = copy.deepcopy(test_dataset.targets)
attack_querries_mask = (attack_querries_mask*0).type(torch.BoolTensor)

attack_querries_index = random.sample(range(0,10000),attack_number)
for index in attack_querries_index:
    attack_querries_mask[index] = True

querries = test_dataset.data.float().unsqueeze(1)/255.

responses = model(querries).detach()

# criterion = nn.CrossEntropyLoss()  # cross entropy loss
criterion = nn.MSELoss()

optimizer = torch.optim.SGD(extracted_model.parameters(), lr=0.1)

for epoch in range(15000):
    optimizer.zero_grad()
    out = extracted_model(querries[attack_querries_mask])
    loss = criterion(out, responses[attack_querries_mask])
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print('number of epoch', epoch, 'loss', loss.data)

predict_out_a = extracted_model(querries)
_, predict_y_a = torch.max(predict_out_a, 1)

predict_out = model(querries)
_, predict_y = torch.max(predict_out, 1)

print('attack accuracy', accuracy_score(predict_y_a.data.numpy(), predict_y.data.numpy()))
# End recording time and memory
end_time = time.time()
end_memory = memory_usage()
# Calculate and print time and memory usage
print('Time taken: ', end_time - start_time, 'seconds')
print('Memory used: ', (end_memory - start_memory) / (1024 * 1024), 'MB')
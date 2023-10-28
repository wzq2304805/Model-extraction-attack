import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

class Net(nn.Module):
    # define nn
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = torch.sigmoid(self.fc1(X))
        X = torch.sigmoid(self.fc2(X))
        X = self.fc3(X)
        X = self.softmax(X)

        return X


# load IRIS dataset
dataset = pd.read_csv('dataset/iris.csv')

# transform species to numerics
dataset.loc[dataset.species == 'Iris-setosa', 'species'] = 0
dataset.loc[dataset.species == 'Iris-versicolor', 'species'] = 1
dataset.loc[dataset.species == 'Iris-virginica', 'species'] = 2

train_X, test_X, train_y, test_y = train_test_split(dataset[dataset.columns[0:4]].values,
                                                    dataset.species.values, test_size=0.8)

# wrap up with Variable in pytorch

train_y = np.array(train_y, dtype=np.long)
test_y = np.array(test_y, dtype=np.long)
train_X = Variable(torch.Tensor(train_X).type(torch.FloatTensor))
test_X = Variable(torch.Tensor(test_X).type(torch.FloatTensor))
train_y = Variable(torch.Tensor(train_y).type(torch.LongTensor))
test_y = Variable(torch.Tensor(test_y).type(torch.LongTensor))

net = Net()

criterion = nn.CrossEntropyLoss()  # cross entropy loss

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

for epoch in range(10000):
    optimizer.zero_grad()
    out = net(train_X)
    loss = criterion(out, train_y)
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print('number of epoch', epoch, 'loss', loss.data)

predict_out = net(test_X)
_, predict_y = torch.max(predict_out, 1)

print('prediction accuracy', accuracy_score(test_y.data, predict_y.data))

torch.save(net.state_dict(), "./multiclass_MLP_new.pt")
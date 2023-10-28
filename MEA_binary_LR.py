# from unittest.test.testmock.support import target
import random


import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

import torch.nn.functional as F


# import and preprocess the dataset
liver_data_train = pd.read_csv('./DataSet/Indian_Liver_Patients_Dataset.csv')

# create a column based on dataset to see if a patient has disease or not
def label_disease(liver_data_train):
    if liver_data_train["Dataset"] == 1:
        return 1
    return 0

liver_data_train['HasDisease'] = liver_data_train.apply(lambda liver_data_train: label_disease(liver_data_train), axis=1)
liver_data_train['Gender'].replace(to_replace=['Male','Female'], value=[0,1],inplace=True)
liver_data_train.isna().sum()
liver_data_train['Albumin_and_Globulin_Ratio'].fillna((liver_data_train['Albumin_and_Globulin_Ratio'].mean()), inplace=True)
Feature = liver_data_train[['Age','Gender','Total_Bilirubin','Direct_Bilirubin',
                      'Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase',
                      'Total_Protiens','Albumin','Albumin_and_Globulin_Ratio']]

X = Feature
y = liver_data_train['HasDisease'].values
X= preprocessing.StandardScaler().fit(X).transform(X)

X_tensor = torch.tensor(X, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.int64)

class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.fc1 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        output = F.sigmoid(x)
        return output

model = LR()


#Define the optimizer for model training
optimizer = optim.SGD(model.parameters(), lr=1e-2) # for LR
def evaluate(model, features, labels):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits
        labels = labels
        indices = torch.where(logits > 0.5, 1, 0).reshape(1, -1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def evaluate_with_response(model, features, labels):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits
        labels = labels
        indices = torch.where(logits > 0.5, 1, 0).reshape(1, -1)
        correct = torch.sum(indices == labels)
        return logits, correct.item() * 1.0 / len(labels)

# target model training
model.train()

criterion = torch.nn.BCELoss()
for epoch in range(3000): #LR 5
    optimizer.zero_grad()
    output = torch.squeeze(model(X_tensor))
    loss  = criterion(output, y_tensor.type(torch.FloatTensor))
    loss.backward()#retain_graph=True
    optimizer.step()

    acc = evaluate(model, X_tensor, y_tensor)
    print('Train Epoch: {} \tLoss: {:.6f} \tAcc: {:.6f}'.format(
        epoch, loss.item(), acc))



model.eval()
test_loss = 0
correct = 0
responses, acc = evaluate_with_response(model, X_tensor, y_tensor)
print('\nTest set: Accuracy: {}\n'.format(acc))

#Attacking the model

#build linear equation

#randomly select d+1 querries
querries = []
querry_index = random.sample(range(583), 11)
querries = X_tensor[querry_index]
response = responses[querry_index]

#generate the equations
b = torch.full([11, 1], 1)
A = torch.cat((querries, b), 1)

B = -torch.log(1/response - 1)

#solve the linear equations
parameters, lu = torch.solve(B,A)
parameters = torch.squeeze(parameters.reshape(1, -1))

#print the result
print("The weights of the target model are:")
print(model.fc1.weight[0])
print("The bias of the target model is ")
print(model.fc1.bias[0])

print("The weights of the extracted model are:")
print(parameters[:10])
print("The bias of the extracted model is ")
print(parameters[10])


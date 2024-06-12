import torch
import torch.nn as nn
from fen_to_matrix import fen_to_matrix
import torch.nn.functional as F
import pickle
import numpy as np

# load data
with open("pos-score-data15k.pkl", "rb") as file:
    pos_score = pickle.load(file)

# extract input and output data
positions = [item[1] for item in pos_score] # shape (10000, 8, 8, 7)
positions = np.array(positions)
positions = torch.tensor(positions, dtype=torch.float32)
scores = [item[2] for item in pos_score]
scores = np.array(scores, dtype=np.float32)
scores = torch.tensor(scores, dtype=torch.float32)

# clean data (get rid of nan values)
nan_indices = torch.isnan(scores)
positions = positions[~nan_indices]
scores = scores[~nan_indices]

# split data into training and testing
train_data = positions[:12000] # originally 12000/15000
train_target = scores[:12000]
test_data = positions[12000:]
test_target = scores[12000:]

class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        
        # Define convolutional layers
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        
        # Define max pooling layer
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        # Define batch normalization layer
        self.bn = nn.BatchNorm3d(64)
        
        # Define fully connected layers
        self.fc1 = nn.Linear(64 * 2 * 2 * 1, 32) # change 3 to 1 for original matrix size
        self.fc2 = nn.Linear(32, 1)
        
        # Define dropout layer
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        # Apply convolutional and pooling layers
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        
        # Apply batch normalization
        x = self.bn(x)
        
        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers and dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
class Net5(nn.Module):
    def __init__(self):
        super(Net5, self).__init__()
        self.linear = nn.Linear(8*8*7, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=3, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(in_channels=3, out_channels=6, kernel_size=3, padding=1)
        self.linear1 = nn.Linear(6*1*2*2, 1)


    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = self.pool(x)
        x = F.tanh(self.conv2(x))
        x = self.pool(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        return x

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(1*4*4*3, 1) # change 6 to 3 for old matrix
    
    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.linear1(x)
        #x = F.relu(x)
        return x

criterion1 = nn.L1Loss()
model5 = Net5()
model2 = Net2()
model3 = Net3()
model4 = Net4()
# load weights
model5.load_state_dict(torch.load("model5_weights.pth"))
model2.load_state_dict(torch.load("model2_weights.pth"))
model3.load_state_dict(torch.load("model3_weights.pth"))
model4.load_state_dict(torch.load("model4_weights.pth"))
# evaluate the model
fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w"
fen2 = "3r3r/1k5p/1p1B2p1/3R4/2K4P/1b6/1b4P1/n7 w - - 2 36"
fen3 = "2krb2r/R5bp/1p1B2p1/1N6/7P/8/PP1K2P1/n7 b - - 4 29"
input1 = torch.tensor(fen_to_matrix(fen1), dtype=torch.float32)
input2 = torch.tensor(fen_to_matrix(fen2), dtype=torch.float32)
input3 = torch.tensor(fen_to_matrix(fen3), dtype=torch.float32)
# for i in range(input.shape[2]):
#     print(input[:,:,i])
# output1 = model4(input1.view(1, 1, 7, 8, 8))
# output2 = model4(input2.view(1, 1, 7, 8, 8))
# output3 = model4(input3.view(1, 1, 7, 8, 8))
# print(output1)
# print(output2)
# print(output3)

with torch.no_grad():
    test_loss = 0
    for i in range(len(test_data)):
        #print(test_data[i].shape, test_target[i].shape)
        output = model4(test_data[i].view(1, 1, 7, 8, 8))
        loss = criterion1(output, test_target[i].view(1, 1))
        test_loss += loss.item()
        if i % 100 == 0:
             #print(test_data[i])
             print(output, test_target[i])
    test_avg_loss = test_loss / len(test_data)
    print(f"Test loss: {test_avg_loss} {len(test_data)} {test_loss}")
# model 2: all outputs are the same, test loss: 208, epoch=5, lr=0.1
# model 3: all outputs are the same, test loss: 205, epoch=5, lr=0.1
# model 4: all outputs are the same, test loss: 205, epoch=5, lr=0.1
# model 5: different outputs, test loss: 204, epoch=5, lr=0.01


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import numpy as np
import matplotlib.pyplot as plt
from fen_to_matrix import fen_to_matrix
from fen_to_matrix import fen_to_matrix2
import matplotlib.pyplot as plt

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

batch_size = 4
trainset = torch.utils.data.TensorDataset(train_data, train_target)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torch.utils.data.TensorDataset(test_data, test_target)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

print(len(positions), len(scores), len(train_data), len(train_target), len(test_data), len(test_target))

# load stockfish data
stockfish_data = []
stockfish_target = []
with open("chessData.csv", "r") as file:
    data = file.readlines()
    stockfish_data = [fen_to_matrix(data.split(",")[0]) for data in data[1:30001]]
    stockfish_target = [float(data.split(",")[1].strip('#')) for data in data[1:30001]]

print(len(stockfish_data), len(stockfish_target))

stockfish_positions = np.array(stockfish_data)
stockfish_positions = torch.from_numpy(stockfish_positions).float()
stockfish_scores = np.array(stockfish_target)
stockfish_scores = torch.from_numpy(stockfish_scores).float()
print(stockfish_positions.shape, stockfish_scores.shape)

nan_indices = torch.isnan(stockfish_scores)
stockfish_positions = stockfish_positions[~nan_indices]
stockfish_scores = stockfish_scores[~nan_indices]

stockfish_train_data = stockfish_positions[:24000]
stockfish_train_target = stockfish_scores[:24000]
stockfish_test_data = stockfish_positions[24000:]
stockfish_test_target = stockfish_scores[24000:]

# define the model: 
# input: 8x8x7 tensor
# output: 1x1x1 tensor
# 2 hidden layer CNN using conv3d 
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=7, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.linear = nn.Linear(64*8*8, 1)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
class Net1_v2(nn.Module): # 1 hidden layer conv2d NN
    def __init__(self):
            super(Net1_v2, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=7, out_channels=16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
            self.linear = nn.Linear(32*8*8, 1)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
class Net1_v3(nn.Module): # no hidden layer conv2d NN
    def __init__(self):
            super(Net1_v3, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=7, out_channels=16, kernel_size=3, padding=1)
            self.linear = nn.Linear(16*8*8, 1)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
# use conv3d instead of conv2d
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

# no hidden layer conv3d net
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
class Net4_v2(nn.Module):
    def __init__(self):
        super(Net4_v2, self).__init__()
        
        # Define convolutional layers
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=2)  # New convolutional layer
        
        # Define max pooling layer
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        # Define batch normalization layer
        self.bn = nn.BatchNorm3d(128)  # Update batch normalization for the new layer
        
        # Define fully connected layers
        self.fc1 = nn.Linear(128 * 2 * 2 * 1, 32) # Change 64 to 128 for the new convolutional layer
        self.fc2 = nn.Linear(32, 1)
        
        # Define dropout layer
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        # Apply convolutional and pooling layers
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))  # New convolutional layer
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
    
# single linear layer
class Net5(nn.Module):
    def __init__(self):
        super(Net5, self).__init__()
        self.linear = nn.Linear(8*8*7, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class Net6(nn.Module):
    def __init__(self):
        super(Net6, self).__init__()
        # Input channels: 7 (6 pieces + 1 color)
        # Output channels: 16
        self.conv1 = nn.Conv2d(7, 16, kernel_size=3, padding=1)
        # Input channels: 16
        # Output channels: 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Input channels: 32
        # Output channels: 64
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers
        self.fc1 = nn.Linear(64, 32)  # 4x4 due to 2 pooling layers
        self.fc2 = nn.Linear(32, 1)  # Output score

    def forward(self, x):
        # Apply convolutional layers followed by ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten the input for the fully connected layers
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# create the model
model = Net1()
model2 = Net2()
model3 = Net3()
model4 = Net4()
model5 = Net5()
model6 = Net6()

model_v2 = Net1_v2()
model_v3 = Net1_v3()
model4_v2 = Net4_v2()
#print(model2(train_data[0].view(1, 1, 7, 8, 8)))

# define loss function and optimizer
criterion0 = nn.MSELoss()
criterion1 = nn.L1Loss()
optim1 = optim.Adam(model.parameters(), lr=0.001)
optim2 = optim.Adam(model2.parameters(), lr=0.01)
optim3 = optim.Adam(model3.parameters(), lr=0.01)
optim4 = optim.Adam(model4.parameters(), lr=0.01)
optim5 = optim.Adam(model5.parameters(), lr=0.01)
optim6 = optim.Adam(model6.parameters(), lr=0.01)

optim1_2 = optim.Adam(model_v2.parameters(), lr=0.01)
optim1_3 = optim.Adam(model_v3.parameters(), lr=0.01)
optim4_2 = optim.Adam(model4_v2.parameters(), lr=0.01)
# MODEL 1
# train_loss_avg = []
# for epoch in range(10):
#     running_loss = 0
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data
#         optim6.zero_grad()
#         outputs = model6(inputs.view(batch_size, 7, 8, 8))
#         loss = criterion1(outputs, labels.view(batch_size, 1))
#         loss.backward()
#         optim6.step()
#         running_loss += loss.item()
#         #print(i)
#     print(f"Epoch {epoch+1}, loss {running_loss}, avg loss {running_loss/(len(trainloader))}")
#     train_loss_avg.append(running_loss/(len(trainloader)))

# print(train_loss_avg)
# torch.save(model5.state_dict(), "model5_weights.pth")

# with torch.no_grad():
#     test_loss = 0
#     for i, data in enumerate(testloader, 0):
#         inputs, labels = data
#         outputs = model6(inputs.view(inputs.shape[0], 7, 8, 8))
#         loss = criterion1(outputs, labels.view(inputs.shape[0], 1))
#         test_loss += loss.item()
#     print(f"Test loss: {test_loss}, avg loss {test_loss/(len(testloader))}")

# plt.clf()
# plt.plot([i+1 for i in range(len(train_loss_avg))], train_loss_avg)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.ylim(0, 200)
# plt.show()

#MODEL 4
# for param in model3.parameters():
#     if param.requires_grad:
#         param.data /= 1000000


train_loss_avg = []
for epoch in range(10):
    running_loss = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optim4.zero_grad()
        outputs = model4(inputs.view(batch_size, 1, 7, 8, 8))
        loss = criterion1(outputs, labels.view(batch_size, 1))
        loss.backward()
        optim4.step()
        running_loss += loss.item()
        #print(i)
    print(f"Epoch {epoch+1}, loss {running_loss}, avg loss {running_loss/(len(trainloader))}")
    train_loss_avg.append(running_loss/(len(trainloader)))

print(train_loss_avg)
torch.save(model4.state_dict(), "model4_weights.pth")

with torch.no_grad():
    test_loss = 0
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        outputs = model4(inputs.view(inputs.shape[0], 1, 7, 8, 8))
        loss = criterion1(outputs, labels.view(inputs.shape[0], 1))
        test_loss += loss.item()
    print(f"Test loss: {test_loss}, avg loss {test_loss/(len(testloader))}")

plt.clf()
plt.plot([i+1 for i in range(len(train_loss_avg))], train_loss_avg)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0, 200)
plt.show()





def conv3d_training(epochs, model, optim, criterion, train_data=train_data, train_target=train_target, test_data=test_data, test_target=test_target, weights_name="model_weights", depth=7):
    #torch.save(model.state_dict(), "model_weights_small.pth")
    train_loss_avg = []
    for epoch in range(epochs):
        running_msg_loss = 0
        avg_cp_loss = 0
        running_score = 0
        for i in range(len(train_data)):
            optim.zero_grad()
            output = model(train_data[i].view(1, 1, depth, 8, 8))
            loss = criterion(output, train_target[i].view(1, 1))
            # if i % 2000 == 0:
            #     print(output, train_target[i].view(1, 1), loss.item())
            loss.backward()
            optim.step()
            running_msg_loss += loss.item()
            running_score += train_target[i].item()
            if i % 2000 == 1999:
                print(f"Epoch {epoch+1}, batch {i+1}: loss {running_msg_loss/2000}, score {running_score/2000}")
                avg_cp_loss += running_msg_loss
                running_msg_loss = 0
                running_score = 0
                # print model weights
        #print(model.conv1.weight)
        avg_cp_loss /= len(train_data)
        train_loss_avg.append(avg_cp_loss)
    print(f"Train loss: {train_loss_avg}")
    # plot training loss across epochs
    plt.plot(train_loss_avg)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")


    with torch.no_grad():
        test_loss = 0
        test_score = 0
        for i in range(len(test_data)):
            output = model(test_data[i].view(1, 1, depth, 8, 8))
            loss = criterion(output, test_target[i].view(1, 1))
            test_loss += loss.item()
            test_score += test_target[i].item()
            # if i % 100 == 0:
            #     print(output)
        test_avg_loss = test_loss / len(test_data)
        print(f"Test loss: {test_avg_loss}, test score: {test_score/2000}")

    # save model weights
    torch.save(model.state_dict(), weights_name)
    return train_loss_avg

def conv3d_test(model, criterion, test_data=test_data, test_target=test_target, depth=7):
    with torch.no_grad():
        test_loss = 0
        for i in range(len(test_data)):
            #print(test_data[i].shape, test_target[i].shape)
            output = model(test_data[i].view(1, 1, depth, 8, 8))
            loss = criterion(output, test_target[i].view(1, 1))
            test_loss += loss.item()
            # if i % 100 == 0:
            #     print(output)
        test_avg_loss = test_loss / len(test_data)
        print(f"Test loss: {test_avg_loss}")

# divide initial weights
i = 0
for param in model6.parameters():
    if param.requires_grad:
        param.data /= 1000000


# conv3d_test(model4, criterion1, test_data, test_target, depth=7)
# train_loss = conv3d_training(10, model4, optim4, criterion1, train_data, train_target, test_data, test_target, weights_name="model4_weights.pth", depth=7)
# plt.clf()
# plt.plot([i+1 for i in range(len(train_loss))], train_loss)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.ylim(0, 205)
# plt.show()


# load weights
# model4.load_state_dict(torch.load("model_weights.pth"))
# # evaluate the model
# input = torch.tensor(fen_to_matrix("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w"), dtype=torch.float32)
# output = model(input.view(1, 1, 7, 8, 8))
# print(output)


# 5 epoch lr 0.01 
#Train loss: [589.8757012873155, 608.1689933798909, 613.3188431802988, 605.1261706346273, 598.44658108747]
# Test loss: 191.27579976431528, test score: 174.4945
def conv2d_training(epochs):
    train_loss_avg = []
    for epoch in range(epochs):
        running_msg_loss = 0
        avg_cp_loss = 0
        running_score = 0
        for i in range(len(train_data)):
            optim6.zero_grad()
            output = model6(train_data[i].view(1, 7, 8, 8))
            loss = criterion1(output, train_target[i].view(1, 1))
            loss.backward()
            optim6.step()
            running_msg_loss += loss.item()
            running_score += train_target[i].item()
            if i % 2000 == 1999:
                print(f"Epoch {epoch+1}, batch {i+1}: loss {running_msg_loss/2000}, score {running_score/2000}")
                avg_cp_loss += running_msg_loss
                running_msg_loss = 0
                running_score = 0
                # print model weights
        #print(model.conv1.weight)
        avg_cp_loss /= len(train_data)
        train_loss_avg.append(avg_cp_loss)
    print(f"Train loss: {train_loss_avg}")

    # plot training loss across epochs
    

    with torch.no_grad():
        test_loss = 0
        test_score = 0
        for i in range(len(test_data)):
            output = model6(test_data[i].view(1, 7, 8, 8))
            loss = criterion1(output, test_target[i].view(1, 1))
            test_loss += loss.item()
            test_score += test_target[i].item()
            # if i % 100 == 0:
            #     print(output)
        test_avg_loss = test_loss / len(test_data)
        print(f"Test loss: {test_avg_loss}, test score: {test_score/2000}")
    torch.save(model.state_dict(), "model6_weights.pth")
    return train_loss_avg

def conv2d_test():
     with torch.no_grad():
        test_loss = 0
        for i in range(len(test_data)):
            #print(test_data[i].shape, test_target[i].shape)
            output = model6(test_data[i].view(1, 7, 8, 8))
            loss = criterion1(output, test_target[i].view(1, 1))
            test_loss += loss.item()
            # if i % 100 == 0:
            #     print(output)
        test_avg_loss = test_loss / len(test_data)
        print(f"Test loss: {test_avg_loss}")


# conv2d_test()
# train_loss = conv2d_training(10)
# plt.clf()
# plt.plot([i+1 for i in range(len(train_loss))], train_loss)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.ylim(0, 200)
# plt.show()

# check for nan values
# for i in range(len(train_data)):
#     if torch.isnan(train_data[i]).any():
#         print("nan values in train_data")
#     if torch.isnan(train_target[i]).any():
#         print("nan values in train_target")
# for i in range(len(test_data)):
#     if torch.isnan(test_data[i]).any():
#         print("nan values in test_data")
#     if torch.isnan(test_target[i]).any():
#         print("nan values in test_target")
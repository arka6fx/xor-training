import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim

#xor dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

Y = np.array([
    [0],
    [1],
    [1],
    [0]
])

#convert data to PyTorch tensors
x_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(Y)

#define the network
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__() #call constructor of the super class module
        self.hidden = nn.Linear(2, 4)
        self.output = nn.Linear(4, 1)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

#create network
torch.manual_seed(42)
model = XORNet()

#loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=2.0)

#Training loop
print("Training PyTorch model")
print("-" * 50)

iterations = 10000

for i in range(iterations + 1):
    #Forward pass
    predictions = model(x_tensor)
    loss = criterion(predictions, y_tensor)

    #Backward pass
        #clear previous gradients
    optimizer.zero_grad()
        #calculate gradients (backpropagation)
    loss.backward()
        #update weights
    optimizer.step()

    if i % 1000 == 0:
        print(f" Iteration: {i} -> Loss: {loss.item()}")


#final predictions
print("\nPytorch Final Predictions : ")
print("-" * 50)
with torch.no_grad():
    final_preds = model(x_tensor)
    print(f"{'Input':^10}{'Target':^10}{'Prediction':^14}{'Rounded':^10}{'Status':^10}")
    for i in range(len(X)):
        rounded = round(final_preds[i].item())
        status = "✅" if rounded == Y[i][0] else "❌"
        print(f"{str(X[i]):^10}{Y[i][0]:^10}{final_preds[i].item():^14.4f}{rounded:^10}{status:^10}")


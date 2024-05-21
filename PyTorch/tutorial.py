import torch
import numpy as np

# Remember: ROW BY COLUMN!!!!

# torch.empty(size): uninitialized
""" x = torch.empty(1) # vector
x = torch.empty(2, 3) # matrix

x = torch.rand(5, 3)
print("rand(5,3):", x)

x = torch.zeros(5, 3) # full of zeros

# check size
print("size", x.size())
print("shape", x.shape)

# check data type
print(x.dtype)

# specify types
x = torch.zeros(5, 3, dtype=torch.float16)
print(x)

# construct from data
x = torch.tensor([5.5, 3])
print(x)

# requires_grad tells pytorch that it will need to calc the gradients for this tensor
# later in your optimization steps
# i.e. this is a var in your model that you want to optimize
# default = false
x = torch.tensor([5.5, 3], requires_grad=True)
print(x)

# Operations
x = torch.ones(2, 2)
y = torch.rand(2, 2)

# elementwise addition
z = x + y
# torch.add(x,y)

# in place addition -> y.add_(x) means it will modify the variable
print(x)
print(y)
print(z)

# also subtraction, mult, div, etc.

# Slicing
x = torch.rand(5, 3)
# x[:, 0] -> all rows, column 0
# x[0, :] -> row 0, all columns
# x[1, 1] -> elm at 1,1

# use .item if only 1 elm in tensor to get actual float value

# Reshape
x = torch.randn(4, 4)
y = x.view(16) # flatten into single row
z = x.view(-1, 8) # size of -1 is inferred from other dimensions
# if -1 then pytorch will auto determine the necessary size in order to have 8 columns

#print(x.size(), y.size(), z.size())

# Convert tensor to numpy arr
a = torch.ones(5)
b = a.numpy()

# Careful: If tensor is on the CPU (not GPU)
# both objects will share the same memory location, so changing one changes the other

# numpy to torch
a = np.ones(5)
b = torch.from_numpy(a) # same memory, changes to a affect b
c = torch.tensor(a) # not same memory

# if GPU is available, move tensors to GPU
device = torch.device('cude' if torch.cuda.is_available() else 'cpu')

x = torch.rand(2,2).to(device) # move tensors to GPU

x = torch.rand(2,2, device=device) # directly create on GPU

# AUTGRAD: algorithm for patial derivatives / chain rule for all operations on Tensors. 
# Computes the vector-Jacobian product
x = torch.randn(3, requires_grad=True)
y = x + 2

# y was created as a result of an operation, so it has a grad_fn attribute
# grad_fn: references a Function that has created the Tensor
print(x) # created by user -> grad_fn is None
print(y)
print(y.grad_fn)

# do more ops on y
z = y * y * 3
print(z)
z = z.mean()
print(z)

# Compute grads with backprop
# .grad attribute becomes the partial derivate of the function w.r.t the tensor

z.backward()
print(x.grad) # dz/dx

# CAREFUL ! ! ! backward() accumulates the gradient with +=
# be sure to do optimizer.zero_grad() for every backprop iteration

# Stop a tensor from tracking history
# useful for when we want to update our weights, or after training during evaluation
# these ops should not be a part of the grad computation
# to prevent, use either of these methods: 
x.requires_grad_(False) # changes the existing flag in place
b = x.detach() # get a new Tensor with the same content but no gradient computation
# wrap in 'with torch.no_grad():'
a = torch.randn(2, 2, requires_grad=True)
with torch.no_grad():
    b = a ** 2 """


# Linear Regression Example: f(x) = w * x + b
# f(x) = 2 * x 

""" X = torch.tensor([1,2,3,4,5,6,7,8], dtype=torch.float32)
Y = torch.tensor([2,4,6,8,10,12,14,16], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model output
def forward(x):
    return w * x

# loss = MSE
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()

X_test = 10.0

print(f'Prediction before training: f({X_test}) = {forward(X_test).item():.3f}')

# Training
learning_rate = 0.01
n_epochs = 100

for epoch in range(n_epochs):
    # predict = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # calc grads = backward pass
    l.backward()

    # update weights
    # w.data = w.data - learning_rate * w.grad
    with torch.no_grad():
        w.sub_(learning_rate * w.grad) # inline subtraction to be safe

    # zero the gradients after updating
    w.grad.zero_()

    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}: w= {w.item():.3f}, loss= {l.item():.3f}')
print(f'Prediction after training: f({X_test}) = {forward(X_test).item():.3f}')  """

# Model, Loss, and Optimizer: Actual PyTorch Pipeline
import torch.nn as nn

# Linear Regression
# f = w * x + b
# here : f = 2 * w

# Step 0.) Training samples, watch the shape!
X = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8], [10], [12], [14], [16]], dtype=torch.float32)

n_samples, n_features = X.shape
print(f'n_samples = {n_samples}, n_features = {n_features}')

# 0.) create a test sample
X_test = torch.tensor([5], dtype=torch.float32)

# 1.) Design the model

# typically would use model = nn.Linear(input_size, output_size)

""" class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()

        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)
    
input_size, output_size = n_features, n_features

model = LinearRegression(input_size, output_size)

print(f'Prediction before training: f({X_test.item()} = {model(X_test).item():.3f})')

# 2.) Define loss and optimizer
learning_rate = 0.01
n_epochs = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3.) Training loop

for epoch in range(n_epochs):
    # predict = forward pass with model
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # backprop
    l.backward()

    # update weights
    optimizer.step()

    # zero grad
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        w, b = model.parameters() # unpack parameters
        print('Epoch ', epoch+1, ': w = ', w[0][0].item(), ' loss = ', l.item())

print(f'Prediction after training: f({X_test.item()} = {model(X_test).item():.3f})') """

# NEURAL NETWORK
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# ONLY FOR GPU STUFF
# device config
device = torch.device('cude' if torch.cuda.is_available() else 'cpu')

# Hypter-parameters
input_size = 784 # 28 x 28
hidden_size = 500
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# Check if the data has already been downloaded
load = not os.path.exists(os.path.join('./PyTorch/data', 'MNIST'))

# MNIST Dataset
train_dataset = torchvision.datasets.MNIST(root='./PyTorch/data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=load)

test_dataset = torchvision.datasets.MNIST(root='./PyTorch/data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

examples = iter(test_loader)
example_data, example_targets = next(examples)

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(example_data[i][0], cmap='gray')
plt.show()


# Fully connected NN with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # Forward pass and loss calc
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backprop and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss{loss.item():.4f}')

# Test the model: we don't need to computer gradients
with torch.no_grad():
    n_correct = 0
    n_samples = len(test_loader.dataset)

    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)

        # max returns (output_value, index)
        _, predicted = torch.max(outputs, 1)
        n_correct += (predicted == labels).sum().item()

    acc = n_correct / n_samples
    print(f'Accuracy of the network on the {n_samples} test images: {100*acc}%')
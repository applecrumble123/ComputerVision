import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import time

"""
# PyTorch replaces arrays in NumPy by Tensors which optimise the computational speed on GPUs
x = torch.rand(5, 3)
print(x)

# Like shape of NumPy, to check the size of a tensor, we can use size method as follow
print(x.size())

# Also like arrays, we can apply operators on tensors. For example, we can add two tensors
y = torch.rand(5, 3)
z=x+ y
print(z)

# convert a tensor to a NumPy array
t = x.numpy()
print(t)

# convert a NumPy array to a tensor
u = torch.from_numpy(t)
print(u)
"""
"""
# N is batch size;
# D_in is input dimension;
# H is hidden dimension;
# D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10


# Create random Tensors to hold inputs and outputs
# N is the batch size
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(torch.nn.Linear(D_in, H),
                            torch.nn.ReLU(),
                            torch.nn.Linear(H, D_out),)

loss_fn = torch.nn.MSELoss(reduction='sum')

# Use the optim package to define an Optimizer that will update the weights of the model for us.
# Here we will use Adam
# The optim package contains many other optimization algorithms.
# The first argument to the Adam constructor tells the # optimizer which Tensors it should update.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train the network 500 times
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss for every time step
    loss = loss_fn(y_pred, y)
    # after every 100 training, plot the error result
    if t % 100 == 99:
        print(t, loss.item())

    # Before the backward pass, use the optimizer object to zero all of the gradients for the variables
    # it will update (which are the learnable # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    # get the gradient and pytorch will calculate the derivative
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model parameters
    # backward propagation
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its parameters
    # update the weights
    optimizer.step()
"""
"""
class TwoLayerNet(torch.nn.Module):
    # D_in: input layer, H: hidden layer, D_out: output layer
    def __init__(self, D_in, H, D_out):
        # In the constructor we instantiate two nn.Linear modules and assign them as member variables.
        
        # call the parent class of the TwoLayerNet
        super(TwoLayerNet, self).__init__()
        # declare 2 linear components
        # input
        # calculate the linear combination of the input with the weights of the layer with the activation function and pass to the hidden layer
        self.linear1 = torch.nn.Linear(D_in, H)
        # output
        # calculate the linear combination of the hidden layer with the weights of the layer with the activation function and pass to the output layer
        self.linear2 = torch.nn.Linear(H, D_out)

    # aims to calculate the response  given an input
    def forward(self, x):
        
        # In the forward function we accept a Tensor of input data and we must return a Tensor of output data.
        # We can use Modules defined in the constructor as well as arbitrary operators on Tensors.
        
        # calculate the linear combination of x and the weights in the input
        # clamp is the relu --> call the activation function
        h_relu = self.linear1(x).clamp(min=0)
        # pass the h_relu in the second component
        y_pred = self.linear2(h_relu)
        return y_pred

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10
# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)
# Construct our loss function and an Optimizer.
# The call to model.parameters() in the SGD constructor will contain the learnable parameters of the two nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)
    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())
        # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
"""

transform = transforms.Compose([transforms.ToTensor(),
                                # (mean of red, mean of green, mean of blue),(std of red, std of green, std of blue)
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

path_train = '/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 5/Task 5.1C/Train Set'
trainset = torchvision.datasets.CIFAR10(root=path_train, train=True, #for training
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=4, #process only 4 images at a time
                                            shuffle=True,
                                            num_workers=2)

path_test = '/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 5/Task 5.1C/Test Set'
testset = torchvision.datasets.CIFAR10(root=path_test,
                                            train=False, #not for training
                                            download=True,
                                            transform=transform)

testloader = torch.utils.data.DataLoader(testset,
                                            batch_size=4, #process only 4 images at a time
                                            shuffle=False,
                                            num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# function to show an image
def imshow(img):
    # normallise --> new_image = (orginal_image - mean) /std
    # unnormalise --> (new_image * std) + mean = orginal_image
    img = img * 0.5 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
# batch size 4
dataiter = iter(trainloader)
images, labels = dataiter.next()
# show images
imshow(torchvision.utils.make_grid(images))
# print true labels of the images
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 3 --> input channels, 6 --> output which is the no. of kernel, 5 --> kernel size
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # 6 --> input from the kernels, 16 --> output which is the no. of kernel, 5 --> kernel size
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 16 * 5 * 5 --> after flattening to get a very long feature vector, input dimension after the 2nd max pooling, 120 --> output dimension
        # 120 --> number of hidden neurons
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # x = self.conv1(x), x = F.relu(x), x = self.pool(x)
        x = self.pool(F.relu(self.conv2(x))) #x = self.conv2(x), x = F.relu(x), x = self.pool(x)
        # the size -1 is inferred from other dimensions, get the no. of rows for 16 * 5 * 5 num of col
        x = x.view(-1, 16 * 5 * 5) #flatten a 16x5x5 tensor to 16x5x5-dimensional vector
        # x --> result after passing the flatten feature vector into fc1
        x = F.relu(self.fc1(x))
        # x --> result after passing the flatten feature vector into fc2, after taking from fc1
        x = F.relu(self.fc2(x))
        # x --> result after passing the flatten feature vector into fc3, after taking from fc2
        x = self.fc3(x)
        return x

# save the class name into the object name (variable)
net = Net()

#loss function --> Classification Cross-Entropy loss
#optimiszer --> Stochastic Gradient Descent (SGD) optimizer, where learning rate (lr) is set to 0.001 and momentum is set to 0.9.

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

start_time = time.time()
loss_history = []

epoch = 2
# 0, 1
for e in range(epoch): # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        # input and true label from training set
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # calculate the loss
        loss = criterion(outputs, labels)
        # back propagation for every 4 images --> see trainloader
        loss.backward()
        # update the weights
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (e + 1, i + 1, running_loss / 2000))
            loss_history.append(running_loss)
            running_loss = 0.0

print('Finished Training')
print("Training time in %s seconds ---" % (time.time() - start_time))

plt.plot(loss_history, label = 'training loss', color = 'r')
plt.legend(loc = "upper left")
plt.show()


PATH = '/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 5/Task 5.1C/cifar_net.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(testloader)
images, labels = dataiter.next()


# print images
imshow(torchvision.utils.make_grid(images))
"""
the % character informs python it will have to substitute something to a token
the s character informs python the token will be a string
the 5 (or whatever number you wish) informs python to pad the string with spaces up to 5 characters.
"""
# the labels is compared to class list to get the class name
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

#load the trained network
net = Net()
# load the weight that was saved
net.load_state_dict(torch.load(PATH))
outputs = net(images)


# torch.max --> returns the maximum value of all elements in the input tensor
# output --> 4 images, output 1 index, which is the class label, with the highest probability value
_, predicted_labels = torch.max(outputs, 1)
"""
the % character informs python it will have to substitute something to a token
the s character informs python the token will be a string
the 5 (or whatever number you wish) informs python to pad the string with spaces up to 5 characters.
"""
print('Predicted: ', ' '.join('%5s' % classes[predicted_labels[j]] for j in range(4)))


start_time = time.time()
correct = 0
total = 0
groundtruth_array = []
predicted_array = []
with torch.no_grad():
    for data in testloader:
        images, groundtruth_labels = data
        for i in groundtruth_labels:
            groundtruth_array.append(i)

        outputs = net(images)
        _, predicted_labels = torch.max(outputs.data, 1)
        for j in predicted_labels:
            predicted_array.append(j)
        total += groundtruth_labels.size(0)
        correct += (predicted_labels == groundtruth_labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
print("Testing time is in %s seconds ---" % (time.time() - start_time))
print("Ground Truth: ",groundtruth_array)
print("Predicted labels: ",predicted_array)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(groundtruth_array, predicted_array)
print(cm)


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, groundtruth_labels = data
        outputs = net(images)
        _, predicted_labels = torch.max(outputs, 1)
        c = (predicted_labels == groundtruth_labels).squeeze()
        for i in range(4):
            label = groundtruth_labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
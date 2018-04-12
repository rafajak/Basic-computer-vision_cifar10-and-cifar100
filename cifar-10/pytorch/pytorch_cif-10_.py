
# coding: utf-8
"""
todo: 
    
    1. Add a helper file printing incorrectly labeled pics in neptune
    2. add image augmentation
    2. Run on gpu 

"""
# In[5]:


# %load pytorch_cif-10.py

import torch
import torchvision
import torchvision.transforms as transforms
from deepsense import neptune
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[6]:


ctx = neptune.Context()
ctx.tags.append('pytorch')
ctx.tags.append('cifar-10')


# In[7]:


epochs = 100
batch_size = 128 # CHANGE TO 128

cuda = torch.cuda.is_available()


# In[8]:


# 1. Loading and normalizing CIFAR10
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[9]:


print(len(trainloader.dataset),
        len(testloader.dataset))


# In[17]:




# # working example 
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
    
#         self.lin = nn.Linear(in_features=32*32*3, out_features = 32)
    
#     def forward(self, x):
#         x = x.view(-1, 32*32*3)
#         x = F.softmax(self.lin(x), dim=-1)

#         return x
    
# model = Net()

# model = Net()

# if cuda: 
#     model.cuda() 


# In[24]:


########################################################################
# 2. Define a Convolution Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
    
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                               kernel_size=3, padding = 1)
        self.conv2 = nn.Conv2d(64, 64, 1, padding = 0)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv4 = nn.Conv2d(128, 128, 1, padding = 0)
        self.conv5 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv6 = nn.Conv2d(256, 256, 1, padding = 0)
        self.conv7 = nn.Conv2d(256, 512, 3, padding = 1)
        self.conv8 = nn.Conv2d(512, 512, 1, padding = 0) # (feature maps are 2pix x 2pix)
        self.fc1 = nn.Linear(512 * 2 * 2, 512) #(num_channels in the last layer * pix_height * pix_width)
        self.fc2 = nn.Linear(512,10)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.dropout06 = nn.Dropout(p=0.6)
        self.dropout05 = nn.Dropout(p=0.5)
        
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.batchnorm4 = nn.BatchNorm2d(512)


    def forward(self, x):
            
        x = F.relu(self.conv1(x))
        x = self.dropout05(self.pool(F.relu(self.conv2(x))))
        
        x = F.relu(self.conv3(x))     
        x = self.dropout06(self.batchnorm2(self.pool(F.relu(self.conv4(x)))))
        
        x = F.relu(self.conv5(x))                  
        x = self.dropout06(self.batchnorm3(self.pool(F.relu(self.conv6(x)))))
        
        x = F.relu(self.conv7(x))     
        x = self.dropout06(self.pool(F.relu(self.conv8(x))))
#         print(x.size())
        x = x.view(-1, 512*2*2) # placeholder (-1) for the first dimension, instead of 'batch_size', which causes issues on the last batch when len(data) % batch_size != 0

        x = self.dropout05(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

model = Net()

if cuda: 
    model.cuda() 


# In[19]:


########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())


# In[20]:


for epoch in range(epochs):

    loss_train_epoch = 0 
    correct_train_epoch = 0
    loss_test_epoch = 0
    correct_test_epoch = 0

    model.train()
    for inputs, labels in trainloader:
        inputs, labels = Variable(inputs), Variable(labels)
        
        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_train_epoch += loss.data[0]
        correct_train_epoch += (outputs.max(1)[1] == labels).sum().data[0]

    avg_loss_train = loss_train_epoch / len(trainloader.dataset)
    avg_acc_train = correct_train_epoch / len(trainloader.dataset)

    model.eval()
    
    for inputs, labels in testloader:
        inputs, labels = Variable(inputs), Variable(labels)

        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss_test_epoch += loss.data[0]
        correct_test_epoch += (outputs.max(1)[1] == labels).sum().data[0]

    avg_loss_test = loss_test_epoch / len(testloader.dataset)
    avg_acc_test = correct_test_epoch / len(testloader.dataset)


    print("Epoch {} \n".format(epoch+1),
    "avg_training_loss: {} \n".format(avg_loss_train),
    "avg_training_acc: {} \n".format(avg_acc_train),
    "avg_test_loss: {} \n".format(avg_loss_test),
    "avg_test_acc: {} \n".format(avg_acc_test))
   
    ctx.channel_send('Loss training', epoch + 1, avg_loss_train)
    ctx.channel_send('Accuracy training', epoch + 1, avg_acc_train)
    ctx.channel_send('Loss test', epoch + 1, avg_loss_test)
    ctx.channel_send('Accuracy test', epoch + 1, avg_acc_test)


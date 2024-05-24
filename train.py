'''

DATA PREPROCESSING

1. Data type conversion
   Convert data from TFRecord format to NumPy arrays for ease of handling
   
2. Make train_dataset, train_loader, test_dataset, test_loader

'''


import preprocessing
from sklearn.model_selection import train_test_split

kface_path = 'kface.tfrecord'
parsed_dataset = preprocessing.get_image_numpy_array(kface_path)

image, label = [], []

for i in range(len(parsed_dataset)):
    image.append(parsed_dataset[i][2])
    label.append(parsed_dataset[i][1][0])



BATCH_SIZE = 64

X_train, X_test, y_train, y_test = train_test_split(image, label, test_size=0.20, random_state=425)

train_dataset = preprocessing.CustomDataset(X_train, y_train)
train_loader = preprocessing.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_dataset = preprocessing.CustomDataset(X_test, y_test)
test_loader =preprocessing.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)


'''

CLASS VGG DEFINITION

1. Model: VGG16 without fully connected layer
2. Functions:
    1. __init__(self, base_dim, dimension):
        - Initialize the VGG model
        
    2. forward(self, x):
        - Define the forward pass of the model
        
    3. initialize(self, module):
        - Function to apply Xavier initialization to the parameters of this model
        
    4. distance(self, x_1, x_2):
        - Function to calculate the weighted sum of the L1 Norm of (x_1 - x_2), using the alpha parameter as weights

'''


import torch
import torchvision
import torch.nn as nn
import torch.functional as F
import torch.nn.init as init

def conv_2(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model

def conv_3(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model


class VGG(nn.Module):
    def __init__(self, dimension, base_dim = 64):
        super(VGG, self).__init__()
        self.feature = nn.Sequential(
            conv_2(3, base_dim),
            conv_2(base_dim, base_dim*2),
            conv_3(base_dim*2, base_dim*4),
            conv_3(base_dim*4, base_dim*8),
            conv_3(base_dim*8, base_dim*8)
        )
        self.alpha = nn.Parameter(torch.Tensor(dimension))
        init.normal_(self.alpha, mean=0.0, std=0.01) 
        
        self.apply(self.initialize)

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, start_dim=1)
     
        return x
    
    def initialize(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0)
    
    def distance(self, x_1, x_2):
        difference = torch.abs(x_1-x_2)
        weighted_sum = torch.sum(self.alpha*difference, dim=-1)    
        prediction = torch.sigmoid(weighted_sum)
        
        return prediction
    
    
'''

Initialize model, hyper_parameter, etc.

'''

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGG(dimension = 4608) 
loss = torch.nn.BCELoss()
optimizer =torch.optim.Adam(model.parameters(), lr = 0.001)

EPOCH = 100
LAMBDA = 0.01 #lambda : [0, 0.1]


'''

REQUIRED FUNCTIONS

1. compute_accuracy_and_loss(model, data_loader, device):
    - Function to compute accuracy and loss for a given model on a given data loader
    
2. bool_to_int(boolean):
    - Function to convert boolean data type to integer (0 or 1)


'''


def compute_accuracy_and_loss(model, data_loader, device):
    accuracy, cost_sum, num_examples = 0, 0, 0
    
    for batch_idx, (image_1, label_1, image_2, label_2) in enumerate(data_loader):
        image_1, image_2 = image_1.to(DEVICE), image_2.to(DEVICE)
        image_1_feature, image_2_feature = model(image_1), model(image_2)
        
        prediction = model.distance(image_1_feature, image_2_feature)

        cost = loss(prediction, bool_to_int(label_1==label_2))
        l2_cost = 0
        for param in model.parameters():
            l2_cost += torch.norm(param, p=2)
        total_cost = cost + l2_cost
        
        num_examples += (label_1==label_2).size(0)
        accuracy += (prediction == (label_1==label_2)).sum()
        cost_sum += total_cost.sum()
        
        print (f'Batch {batch_idx:03d}/{len(data_loader):03d} |'
               f' Cost: {total_cost.mean():.4f}')
        
    return accuracy/num_examples * 100, cost_sum/num_examples


def bool_to_int(boolean):
    target = [1 if b else 0 for b in boolean]
    
    return torch.tensor(target).float()


'''

TRAINING

'''

import time

start_time = time.time()
train_acc_lst, train_loss_lst, test_acc_lst, test_loss_lst = [], [], [], []

model.to(DEVICE)

for epoch in range(EPOCH):
    model.train()

    for batch_idx, (image_1, label_1, image_2, label_2) in enumerate(train_loader):
        image_1, image_2 = image_1.to(DEVICE), image_2.to(DEVICE)
        image_1_feature, image_2_feature = model(image_1), model(image_2)

        prediction = model.distance(image_1_feature, image_2_feature)

        cost = loss(prediction, bool_to_int(label_1==label_2))
        l2_cost = 0
        for param in model.parameters():
            l2_cost += torch.norm(param, p=2)
        total_cost = cost + l2_cost
        
        optimizer.zero_grad()
        total_cost.backward()
        optimizer.step()
        
        print (f'Epoch: {epoch:03d} | '
               f'Batch {batch_idx:03d}/{len(train_loader):03d} |'
               f'Cost: {total_cost:.4f}')
        
    model.eval()
    
    train_acc, train_loss = compute_accuracy_and_loss(model, train_loader, device=DEVICE)
    test_acc, test_loss = compute_accuracy_and_loss(model, test_loader, device=DEVICE)
    train_acc_lst.append(train_acc)
    test_acc_lst.append(test_acc)
    
    print(f'Epoch: {epoch:03d}/{EPOCH:03d} Train Acc.: {train_acc:.2f}%'
          f' | Test Acc.: {test_acc:.2f}%')

    elapsed = (time.time() - start_time)/60
    print(f'Time elapsed: {elapsed:.2f} min')

elapsed = (time.time() - start_time)/60
print(f'Total Training Time: {elapsed:.2f} min')


import os

current_dir = os.getcwd()
model_save_path = os.path.join(current_dir, 'vgg_model.pth')
torch.save(model.state_dict(), model_save_path)

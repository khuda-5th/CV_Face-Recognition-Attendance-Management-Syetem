#####################################################################################
#                                                                                   #
# Model Name : Siampain                                                             #
# Model Purpose : Face Recognition                                                  #
# Backbone : VGG-16                                                                 #
# Reference : Siamese Neural Networks for One-shot Image Recognition                #
# Library : Pytorch                                                                 #
#                                                                                   #
# siampain_train.py & Preprocessing.py written by hayoung Lee(lhayoung9@khu.ac.kr)  #
#                                                                                   #
# Additional Information                                                            #
# - Dataset : AI Hub - 마스크 착용 한국인 안면 이미지 데이터                          #
#                                                                                   #
#####################################################################################

'''

DATA PREPROCESSING                                                                
                                                       
1. Converting Data type 
   - Converting data from tfrecord to numpy array for use with PyTorch

2. Making train_dataset, train_loader, test_dataset, test_loader

'''


import preprocessing
from sklearn.model_selection import train_test_split

kface_path = 'kface.tfrecord'
parsed_dataset = preprocessing.get_image_numpy_array(kface_path)

image, label = [], []

for i in range(len(parsed_dataset)):
    image.append(parsed_dataset[i][2])
    label.append(parsed_dataset[i][1][0])

X_train, X_test, y_train, y_test = train_test_split(image, label, test_size=0.20, random_state=425)

BATCH_SIZE = 16

train_dataset = preprocessing.CustomDataset(X_train, y_train)
train_loader = preprocessing.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_dataset = preprocessing.CustomDataset(X_test, y_test)
test_loader =preprocessing.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)


'''

VGG CLASS DEFINITION

1. Backbone : Pre-trained VGG-16 without fully connected layer
2. Functions:
    1. __init__(self, base_dim, dimension):
        - Initialize the VGG model
        - Initialize parameter alpha according to normal distribution N(0, 0.1)
        - Arges:
            - base_dim : Skip
            - dimension : Flattened tensor shape after passing through the forward pass

    2. forward(self, x):
        - Define the forward pass of the model
        - Args:
            - x: Input tensor to the model
        - Returns:
            - Output tensor after forward pass

    3. distance(self, x_1, x_2):
        - Calculate the weighted sum of the L1 Norm of (x _1 - x_2), using the alpha parameter as weights
        - Weighted sum of L1 Norm after passing through a sigmoid function

'''


import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
import torch.functional as F

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

    def distance(self, x_1, x_2):
        difference = torch.abs(x_1-x_2)
        weighted_sum = torch.sum(self.alpha*difference, dim=-1)

        prediction = torch.sigmoid(weighted_sum)

        return prediction


'''

MODEL DECLARATION, HYPERPARAMETER INITIALIZATION, ETC.

Using Pre-trained VGG-16 without fully connected layer
Loss function: Binary Cross Entropy
Optimizer: Adam with learning rate set to 0.001
EPOCHS: 10
    - I set EPOCHS to 10 based on experimental results, where the accuracy on the test set reached 95% after 7 epochs

'''

import torchvision.models as models


# Define the custom VGG model with 4608 dimension

model = VGG(dimension = 4608)

# Set the device to GPU if available, otherwise use CPU

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# code for transfer Learning 
# Transfer the weights from pre-trained VGG-16 to the custom model

vgg16 = models.vgg16(pretrained=True)
vgg16_features = list(vgg16.features.children())

conv_layers = [layer for layer in model.feature.modules() if isinstance(layer, nn.Conv2d)]
vgg16_conv_layers = [layer for layer in vgg16_features if isinstance(layer, nn.Conv2d)]

for my_layer, vgg16_layer in zip(conv_layers, vgg16_conv_layers):
    my_layer.weight.data = vgg16_layer.weight.data.clone()
    if my_layer.bias is not None:
        my_layer.bias.data = vgg16_layer.bias.data.clone()
        

# Setting

EPOCH = 5
loss = torch.nn.BCELoss()
optimizer =torch.optim.Adam(model.parameters(), lr = 0.001)


'''

UTILITY FUNCTIONS

1. compute_accuracy_and_loss(model, data_loader, device):
    - Compute accuracy and loss for a given model on a given data loader
    - Args : 
        - model : The neural network model to be evaluated
        - data_loader : DataLoader object providing the dataset for evaluation
        - device : Device to be used for computation (e.g., 'cuda' or 'cpu')

2. bool_to_int(boolean):
    - Convert tensor of boolean data type to tensor of integer data type
    - Args : 
        - boolean : Tensor of boolean data type
    - Returns : 
        - Tensor of integer data type

'''


def compute_accuracy_and_loss(model, data_loader, device):
    accuracy, cost_sum, num_samples = 0, 0, 0

    for batch_idx, (image_1, label_1, image_2, label_2) in enumerate(data_loader):
        image_1, image_2 = image_1.to(DEVICE), image_2.to(DEVICE)
        image_1_feature, image_2_feature = model(image_1), model(image_2)

        predicted_similarity = model.distance(image_1_feature, image_2_feature)
        
        # If the predicted similarity is greater than 0.5, the scaled predicted similarity is set to 1
        # If the predicted similarity is less than or equal to 0.5, the scaled predicted similarity is set to 0
        
        scaled_predicted_similarity = [1 if predicted_similarity[i]  > 0.5 else 0 for i in range(len(predicted_similarity))]

        # Calculate loss using Binary Cross Entropy

        cost = loss(scaled_predicted_similarity, bool_to_int(label_1==label_2))

        
        num_samples += (label_1==label_2).size(0)
        
        # If the scaled predicted similarity matches the target, increase accuracy by 1
        
        accuracy += (torch.tensor(scaled_predicted_similarity) == bool_to_int(label_1==label_2)).sum()

        cost_sum += cost.sum()

        print (f'Batch {batch_idx:03d}/{len(data_loader):03d} |'
               f' Cost: {cost:.4f}')

    return accuracy/num_samples * 100, cost_sum/num_samples


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
        
        # Calculate loss using Binary Cross Entropy
        
        cost = loss(prediction, bool_to_int(label_1==label_2))

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print (f'Epoch: {epoch:03d} | '
               f'Batch {batch_idx:03d}/{len(train_loader):03d} |'
               f'Cost: {cost:.4f}')

    model.eval()
    
    with torch.no_grad():
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
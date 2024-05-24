'''

Data preprocessing

1. Data type : tfrecord -> numpy 
2. make train_dataset, train_loader, test_dataset, test_loader

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

print(y_train)

train_dataset = preprocessing.CustomDataset(X_train, y_train)
train_loader = preprocessing.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_dataset = preprocessing.CustomDataset(X_test, y_test)
test_loader =preprocessing.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)


'''

Model, Loss function Definition

'''

import torch
import torchvision
import torch.nn as nn
import torch.functional as F

from Contrastive_Loss import Contrastive_Loss


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.vgg16(weights='DEFAULT') #Use Pre-trained VGG19 
loss = Contrastive_Loss(margin = 1)
optimizer =torch.optim.Adam(model.parameters(), lr = 0.001)

EPOCH = 100
BATCH_SIZE = 64


'''

'compute_accuracy_and_loss' function Definition 

'''

def compute_accuracy_and_loss(model, data_loader, device):
    accuracy, loss_sum, num_examples = 0, 0, 0
    for batch_idx, (image_1, label_1, image_2, label_2) in enumerate(data_loader):
        image_1 = image_1.to(device)
        image_1_feature = model(image_1)
        
        image_2 = image_2.to(device)
        image_2_feature = model(image_2)
        
        Loss, predict = loss(image_1_feature, image_2_feature, label_1 == label_2)
        
        num_examples += (label_1==label_2).size(0)
        accuracy += (predict == (label_1==label_2)).sum()
        loss_sum += Loss.sum()
        
        print (f'Batch {batch_idx:03d}/{len(data_loader):03d} |'
               f' Cost: {Loss.mean():.4f}')
        
    return accuracy/num_examples * 100, loss_sum/num_examples


'''

Training

'''

import time

start_time = time.time()
train_acc_lst, train_loss_lst, test_acc_lst, test_loss_lst = [], [], [], []

model.to(DEVICE)

for epoch in range(EPOCH):
    model.train()

    for batch_idx, (image_1, label_1, image_2, label_2) in enumerate(train_loader):
        image_1 = image_1.to(DEVICE)
        image_1_feature = model(image_1)
        
        image_2 = image_2.to(DEVICE)
        image_2_feature = model(image_2)

        Loss, _ = loss(image_1_feature, image_2_feature, label_1 == label_2)
        
        Loss = Loss.mean()
        
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
        
        print (f'Epoch: {epoch:03d} | '
               f'Batch {batch_idx:03d}/{len(train_loader):03d} |'
               f' Cost: {Loss:.4f}')
        
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

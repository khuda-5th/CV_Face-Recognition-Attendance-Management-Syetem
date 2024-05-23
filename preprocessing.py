import torch
import numpy as np
from tfrecord.torch.dataset import TFRecordDataset
import matplotlib.pyplot as plt
import cv2  # OpenCV를 사용하여 이미지를 디코딩합니다.

def parse_tfrecord(example):
    filename = example['image/filename'].decode('utf-8')  # byte to string
    source_id = example['image/source_id']
    encoded_image = example['image/encoded']
    
    image_array = np.frombuffer(encoded_image, dtype=np.uint8)
    image_array = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    return filename, source_id, image_array

def get_image_numpy_array(tfrecord_path):
    description = {
        'image/filename': 'byte',
        'image/source_id': 'int',
        'image/encoded': 'byte'
    }
    
    dataset = TFRecordDataset(tfrecord_path, index_path=None, description=description)
    parsed_dataset = [parse_tfrecord(example) for example in dataset]

    return parsed_dataset

import os
import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, image_array, label, transform=None):
        self.image_array = image_array
        self.labels = label

    def __len__(self):
        return len(self.image_array)
    
    def center(self, image):
        mean = torch.mean(image, dim=(1, 2), keepdim=True)
        centered_image = image - mean

        return centered_image

    def transform(self, image_array):
        image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        image = transforms.ToTensor()(image)
        image = self.center(image)

        return image

    def __getitem__(self, idx):
        length = self.__len__()
        
        image_1 = self.transform(self.image_array[idx])
        label_1 = self.labels[idx]
        
        image_2 = self.transform(self.image_array[(idx+1)%length])
        label_2 = self.labels[(idx+1)%length]

        return image_1, label_1, image_2, label_2
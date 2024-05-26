import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vgg16
from PIL import Image, ImageDraw, ImageFont
import json
import os
from facenet_pytorch import MTCNN
import time
import sys

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoding_file = './encodings.json'

# Face recognition model class
class FaceRecognitionModel(nn.Module):
    def __init__(self):
        super(FaceRecognitionModel, self).__init__()
        self.model = vgg16(pretrained=False)
        self.model.classifier = nn.Sequential(
            *list(self.model.classifier.children())[:-1]
        )

    def forward(self, x):
        return self.model(x)

    def load_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("features.", "model.features.")
            new_key = new_key.replace("classifier.", "model.classifier.")
            if "model.classifier.6" in new_key:
                continue
            new_state_dict[new_key] = value

        super(FaceRecognitionModel, self).load_state_dict(new_state_dict)

# 얼굴 인식 함수
def recognize_face(model, face, known_face_encodings, known_face_ids):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    face = Image.fromarray(face)
    face = transform(face).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        face_encoding = model(face).cpu().numpy().flatten()

    if len(known_face_encodings) == 0:
        return "New Face", face_encoding

    matches = np.linalg.norm(known_face_encodings - face_encoding, axis=1)

    if len(matches) > 0:
        min_distance_index = np.argmin(matches)
        if matches[min_distance_index] < 0.6:  # 임계값
            student_id = known_face_ids[min_distance_index]
            return student_id, face_encoding
        
    return "New Face", face_encoding

def save_known_faces(ids, encodings, names):
    data = [[id_, name, encoding] for id_, encoding, name in zip(ids, encodings, names)]
    with open(encoding_file, 'w') as f:
        json.dump(data, f)

    # Restart the Streamlit session
    # os.execl(sys.executable, sys.executable, *sys.argv)

def load_known_faces():
    if os.path.exists(encoding_file):
        with open(encoding_file, 'r') as f:
            data = json.load(f)
            known_face_ids = [entry[0] for entry in data]
            known_face_names = [entry[1] for entry in data]
            known_face_encodings = [entry[2] for entry in data]
            print("load_known_faces()에 잇는 프린트: ", known_face_names)
            print("load_known_faces()에 잇는 프린트 수: ", len(known_face_names))
        
        return known_face_ids, known_face_encodings, known_face_names
    else:
        return [], [], []

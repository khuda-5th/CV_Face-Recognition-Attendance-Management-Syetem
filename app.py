import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vgg16
from PIL import Image
from facenet_pytorch import MTCNN

# 얼굴 인식 모델 클래스
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
        # 로드할 때 사용되는 키 이름을 수정하여 일치시킵니다.
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("features.", "model.features.")  
            new_key = new_key.replace("classifier.", "model.classifier.")
            if "model.classifier.6" in new_key:
                continue
            new_state_dict[new_key] = value
            
        super(FaceRecognitionModel, self).load_state_dict(new_state_dict)


# 얼굴 인식 함수
known_face_encodings = []  # 초기화

known_face_encodings = []  # 초기화
known_face_names = []      # 초기화

def recognize_face(model, face, known_face_encodings, known_face_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    face = Image.fromarray(face)
    face = transform(face).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        face_encoding = model(face).cpu().numpy().flatten()
    
    # known_face_encodings와 known_face_names에 새로운 얼굴 인코딩과 이름 추가
    known_face_encodings.append(face_encoding)
    known_face_names.append("New Face")

    matches = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
    name = "Unknown"
    
    if len(matches) > 0:
        min_distance_index = np.argmin(matches)
        if matches[min_distance_index] < 0.6:  # 임계값
            name = known_face_names[min_distance_index]
    
    return name

# Streamlit 앱 메인 함수
def main():
    st.title("📸 AI 얼굴인식 출결관리 시스템 📸")
    run = st.checkbox('웹캠 시작/정지')

    # 사이드바
    st.sidebar.subheader("CV트랙 출석부")

    FRAME_WINDOW = st.image([])

    # MTCNN 로드
    mtcnn = MTCNN(keep_all=True, device=DEVICE)

    camera = cv2.VideoCapture(0)

    # 모델 로드
    model_save_path = './model_pth/best_model.pth'
    model = FaceRecognitionModel().to(DEVICE)
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))

    model.eval()

    # 알려진 얼굴 인코딩 및 이름 (이 예제에서는 비어 있음)
    known_face_encodings = []  # numpy 배열로 저장된 인코딩 리스트
    known_face_names = []      # 인코딩에 대응되는 이름 리스트

    while run:
        ret, frame = camera.read()
        if not ret:
            st.write("카메라를 찾을 수 없습니다.")
            break
        
        # 좌우 반전
        frame = cv2.flip(frame, 1)

        # 얼굴 검출
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb_frame)
        
        # 얼굴에 사각형 그리기 및 텍스트 추가
        if boxes is not None:
            for box in boxes:
                x, y, w, h = map(int, box)
                face = frame[y:h, x:w]
                name = recognize_face(model, face, known_face_encodings, known_face_names)
                cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # BGR에서 RGB로 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    camera.release()

if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()

import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vgg16
from PIL import Image, ImageDraw, ImageFont

from facenet_pytorch import MTCNN
import time

# 나눔고딕 폰트 경로
font_path = 'NanumGothicBold.ttf'

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
known_face_names = []  # 초기화


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

    if len(known_face_encodings) == 0:
        return "New Face", face_encoding

    matches = np.linalg.norm(known_face_encodings - face_encoding, axis=1)

    if len(matches) > 0:
        min_distance_index = np.argmin(matches)
        if matches[min_distance_index] < 0.6:  # 임계값
            name = known_face_names[min_distance_index]
            return name, face_encoding

    return "New Face", face_encoding


# Streamlit 앱 메인 함수
def main():
    st.title("📸 AI 얼굴인식 출결관리 시스템 📸")

    # 사이드바
    st.sidebar.subheader("📝 CV트랙 출석부")
    st.sidebar.checkbox('김민권')
    st.sidebar.checkbox('도윤서')
    st.sidebar.checkbox('류여진')
    st.sidebar.checkbox('박현준')
    st.sidebar.checkbox('이하영')
    st.sidebar.checkbox('임성은')
    st.sidebar.checkbox('장서연')

    st.sidebar.markdown("-------------------")
    st.sidebar.subheader("🚪 출튀 명단")
    st.sidebar.write("1. 임성은")
    st.sidebar.write("2. 박현준")

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
    known_face_names = []  # 인코딩에 대응되는 이름 리스트

    registering = False
    new_face_encoding = None

    form_key_suffix = 0  # 고유한 폼 키s를 만들기 위한 숫자

    while True:
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
                name, face_encoding = recognize_face(model, face, known_face_encodings, known_face_names)
                cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)
                

                # 한글 폰트 설정
                font = ImageFont.truetype(font_path, 24)
                frame_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(frame_pil)
                draw.text((x, y - 10), name, font=font, fill=(255, 255, 255))
                frame = np.array(frame_pil)


                if name == "New Face" and not registering:
                    registering = True
                    new_face_encoding = face_encoding
                    st.write("처음 오셨군요! 등록을 진행합니다.")
                    with st.form(f"register_form_{form_key_suffix}"):
                        student_id = st.text_input("학번")
                        student_name = st.text_input("이름")
                        submit = st.form_submit_button("제출")
                        if submit:
                            known_face_encodings.append(new_face_encoding)
                            known_face_names.append(student_name)
                            registering = False
                            success_message = st.success(f"{student_name} 님, 등록이 완료되었습니다!")
                            time.sleep(3)  # 3초간 메시지 표시
                            success_message.empty()  # 메시지 제거
                            form_key_suffix += 1

        # BGR에서 RGB로 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    camera.release()


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()

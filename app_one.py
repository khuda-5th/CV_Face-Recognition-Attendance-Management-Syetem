import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import json
from facenet_pytorch import MTCNN
import time
from recognize_face import * 

# Paths
font_path = 'NanumGothicBold.ttf'
encoding_file = 'encodings.json'

# Main Streamlit app function
def main():

    # Initialize known face variables
    known_face_ids, known_face_encodings, known_face_names = [], [], []

    st.title("📸 AI 얼굴인식 출결관리 시스템 📸")

    # Sidebar
    st.sidebar.subheader("📝 CV트랙 출석부")

    FRAME_WINDOW = st.image([])

    # MTCNN loading
    mtcnn = MTCNN(keep_all=True, device=DEVICE)

    camera = cv2.VideoCapture(0)

    # Model loading
    model_save_path = './model_pth/best_model.pth'
    model = FaceRecognitionModel().to(DEVICE)
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
    model.eval()

    registering = False
    new_face_encoding = None
    form_key_suffix = 0
    form = None

    while True:
        ret, frame = camera.read()
        if not ret:
            st.write("카메라를 찾을 수 없습니다.")
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb_frame)

        if boxes is not None:
            for box in boxes:
                x, y, w, h = map(int, box)
                face = frame[y:h, x:w]
                name, face_encoding = recognize_face(model, face, known_face_encodings, known_face_names)
                cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)
                
                font = ImageFont.truetype(font_path, 24)
                frame_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(frame_pil)
                draw.text((x, y - 10), name, font=font, fill=(255, 255, 255))
                frame = np.array(frame_pil)

                if name == "New Face" and not registering:
                    registering = True
                    new_face_encoding = face_encoding
                    st.write("처음 오셨군요! 등록을 진행합니다.")
                    form = st.form(f"register_form_{form_key_suffix}")
                    student_id = form.text_input("학번")
                    student_name = form.text_input("이름")
                    submit = form.form_submit_button("제출")
                    if submit:
                        known_face_ids.append(student_id)
                        known_face_encodings.append(new_face_encoding.tolist())
                        known_face_names.append(student_name)
                        save_known_faces(known_face_ids, known_face_encodings, known_face_names)
                        registering = False
                        success_message = st.success(f"{student_name} 님, 등록이 완료되었습니다!")
                        time.sleep(3)
                        success_message.empty()
                        form_key_suffix += 1

                        # Reload known faces data after registration
                        known_face_ids, known_face_encodings, known_face_names = load_known_faces()

                        print("등록 사람 수: ", len(known_face_names))
                        print("등록 사람 이름: ", known_face_names)
                        if len(known_face_names) != 0:
                            for student_name in known_face_names:
                                st.sidebar.checkbox(student_name)

                        st.sidebar.markdown("-------------------")
                        st.sidebar.subheader("🚪 출튀 명단")
                        #출튀 명단이 0이 아니면 아래에 이름 작성

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    camera.release()

if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    known_face_ids, known_face_encodings, known_face_names = load_known_faces()
    main()
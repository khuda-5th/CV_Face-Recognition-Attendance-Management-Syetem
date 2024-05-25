import os
import shutil

def move_images_with_folders(source_dir, destination_dir):
    """
    특정 디렉토리 안에 있는 jpg 파일들을 다른 디렉토리로 옮기는데,
    각 이미지 파일마다 옮겨지는 디렉토리에서 이미지 파일과 이름이 동일한 폴더를 만들어 그 안에 저장합니다.

    Args:
    source_dir (str): 원본 이미지 파일들이 있는 디렉토리 경로
    destination_dir (str): 이미지를 옮길 대상 디렉토리 경로

    Returns:
    None
    """
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for filename in os.listdir(source_dir):
        if filename.lower().endswith('.jpg'):
            # 원본 이미지 파일 경로
            source_path = os.path.join(source_dir, filename)
            
            # 대상 폴더 경로 (이미지 파일과 동일한 이름의 폴더)
            folder_name = os.path.splitext(filename)[0]
            target_folder = os.path.join(destination_dir, folder_name)
            
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            
            # 대상 이미지 파일 경로
            target_path = os.path.join(target_folder, filename)
            
            # 이미지 파일 이동
            shutil.move(source_path, target_path)
            print(f'Moved {source_path} to {target_path}')

# 예제 사용
source_directory = 'preprocessing/sample_data_resize'  # 원본 이미지 파일들이 있는 디렉토리 경로
destination_directory = 'preprocessing/data/k-face/old_path'  # 이미지를 옮길 대상 디렉토리 경로

move_images_with_folders(source_directory, destination_directory)

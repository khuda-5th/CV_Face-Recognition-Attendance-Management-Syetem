import os

# 디렉토리가 있는 폴더 경로
folder_path = 'preprocessing/data/VS1_resize'

# 폴더 안에 있는 디렉토리 이름 가져오기
dir_list = os.listdir(folder_path)

# 디렉토리 이름을 2700부터 2749까지 변경
for i, dir_name in enumerate(dir_list):
    new_dir_name = str(2750 + i)
    os.rename(os.path.join(folder_path, dir_name), os.path.join(folder_path, new_dir_name))

print("디렉토리 이름 변경 완료!")
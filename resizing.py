import os
from PIL import Image

def resize_image(image_path, output_path, size=(800, 800)):
    with Image.open(image_path) as img:
        img = img.resize(size, Image.LANCZOS)
        img.save(output_path)
        
def resize_all_images(cur_dir, resize_folder_path, size=(112, 112)):
    for filename in os.listdir(cur_dir):
        if filename.lower().endswith('jpg'):
            file_path = os.path.join(cur_dir, filename)
            output_path = os.path.join(resize_folder_path, filename)
            resize_image(file_path, output_path, size)
            print(f"Resized {file_path} and saved as {output_path}")
            
def REI1(cur_dir, resize_folder_path, size):
    for foldername in os.listdir(cur_dir):  # cur_dir : REI1
        if os.path.isdir(os.path.join(cur_dir, foldername)):
            work_dir = os.path.join(cur_dir, foldername)  # work_dir : REI1/1_0_00_0_01
            for num in os.listdir(work_dir):
                if os.path.isdir(os.path.join(work_dir, num)):
                    img_dir = os.path.join(work_dir, num)  # img_dir : REI1/1_0_00_0_01/1
                    resize_all_images(img_dir, resize_folder_path, size)
                    
def SPI(cur_dir, resize_folder_path, size):
    if os.path.isdir(os.path.join(cur_dir, '3D')):
        img_dir = os.path.join(cur_dir, '3D')
        resize_all_images(img_dir, resize_folder_path, size)
                    
def STD2(cur_dir, resize_folder_path, size):
    for foldername in os.listdir(cur_dir):  # cur_dir : STD2
        if os.path.isdir(os.path.join(cur_dir, foldername)):
            work_dir = os.path.join(cur_dir, foldername)  # work_dir : STD2/1_0_00_0_01
            if os.path.isdir(os.path.join(work_dir, 'RGB')):  # IR 제외 RGB만
                img_dir = os.path.join(work_dir, 'RGB')  # img_dir : STD2/1_0_00_0_01/RGB
                resize_all_images(img_dir, resize_folder_path, size)

def resize_images_in_folders(base_dir, resize_dir, start=2400, end=2699, size=(112, 112)):
    if not os.path.exists(resize_dir):
        os.makedirs(resize_dir)
    
    for foldername in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, foldername)
        resize_folder_path = os.path.join(resize_dir, foldername)
        
        if os.path.isdir(folder_path):
            
            if not os.path.exists(resize_folder_path):
                os.makedirs(resize_folder_path)
                
            # folder_path : 1007025
            if os.path.isdir(os.path.join(folder_path, 'REI1')):
                cur_dir = os.path.join(folder_path, 'REI1')  # cur_dir : 1007025/REI1
                REI1(cur_dir, resize_folder_path, size)
                
            if os.path.isdir(os.path.join(folder_path, 'SPI')):
                cur_dir = os.path.join(folder_path, 'SPI')  # cur_dir : 1007025/SPI
                SPI(cur_dir, resize_folder_path, size)
                
            if os.path.isdir(os.path.join(folder_path, 'STD2')):
                cur_dir = os.path.join(folder_path, 'STD2')  # cur_dir : 1007025/STD2
                STD2(cur_dir, resize_folder_path, size)


base_directory = 'preprocessing/VS1-2'  
resize_directory = 'preprocessing/VS1-2_resize'  
resize_images_in_folders(base_directory, resize_directory)
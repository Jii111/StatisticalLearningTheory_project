# based on https://colab.research.google.com/github/roboflow-ai/yolov5-custom-training-tutorial/blob/main/yolov5-custom-training.ipynb?authuser=1#scrollTo=yNveqeA1KXGy

## clone YOLOv5
!git clone https://github.com/ultralytics/yolov5 
%cd yolov5
%pip install -qr requirements.txt # install dependencies
%pip install -q roboflow

import torch
import os
from IPython.display import Image, clear_output

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

## check image
import os
import glob

def count_images_in_folder(folder_path, image_extensions=['.jpg', '.jpeg', '.png']):
    # 폴더 내의 이미지 파일 목록 가져오기
    image_files = []
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, '*' + extension)))

    # 이미지 파일 수 반환
    return len(image_files)

# 폴더 경로 설정
folder_path = '/content/yolov5/1129train-2/train/images'
# 이미지 수 세기
image_count = count_images_in_folder(folder_path)
# 결과 출력
print(f'Total number of images in the folder: {image_count}')

# 폴더 경로 설정
folder_path = '/content/yolov5/1129validd-1/train/images'
# 이미지 수 세기
image_count = count_images_in_folder(folder_path)
# 결과 출력
print(f'Total number of images in the folder: {image_count}')

## training
!python train.py --batch 4 --epochs 32 --cfg /content/yolov5s_custom.yaml  --data /content/cococustom_yolov5.yaml --weights /content/yolov5s.pt --cache

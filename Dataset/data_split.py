import os
from shutil import copyfile
import random

def split_and_match_annotations(image_folder, output_folder, train_ratio=0.7, test_ratio=0.15, valid_ratio=0.15):

    # 폴더 생성
    train_folder = os.path.join(output_folder, 'train')
    test_folder = os.path.join(output_folder, 'test')
    valid_folder = os.path.join(output_folder, 'valid')

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(valid_folder, exist_ok=True)

    # 이미지 파일 리스트 가져오기
    images = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # 데이터셋 분할
    images.sort()
    random.shuffle(images)
    num_images = len(images)

    train_end = int(train_ratio * num_images)
    test_end = train_end + int(test_ratio * num_images)

    train_set = images[:train_end]
    test_set = images[train_end:test_end]
    valid_set = images[test_end:]

    # 이미지 파일과 대응되는 annotation 파일 복사
    for img in train_set:
        copyfile(os.path.join(image_folder, img), os.path.join(train_folder, img))
        
    for img in test_set:
        copyfile(os.path.join(image_folder, img), os.path.join(test_folder, img))
        
    for img in valid_set:
        copyfile(os.path.join(image_folder, img), os.path.join(valid_folder, img))
        
    print(len(train_set))
    print(len(test_set))
    print(len(valid_set))

# 이미지 파일을 훈련, 테스트, 검증 세트로 나누기
image_folder="C:/Users/judy0/train/images"
output_folder="C:/Users/judy0/newdata"
split_and_match_annotations(image_folder, output_folder)

# 훈련, 테스트, 검증 파일 리스트 불러오기
train_set = set(os.listdir(os.path.join(output_folder, 'train')))
test_set = set(os.listdir(os.path.join(output_folder, 'test')))
valid_set = set(os.listdir(os.path.join(output_folder, 'valid')))

def separate_annotations_by_image_names(image_folder, annotation_folder, output_folder, train_set, test_set, valid_set):
    # 폴더 생성
    output_train_folder = os.path.join(output_folder, 'train')
    output_test_folder = os.path.join(output_folder, 'test')
    output_valid_folder = os.path.join(output_folder, 'valid')

    os.makedirs(output_train_folder,exist_ok=True)
    os.makedirs(output_test_folder,exist_ok=True)
    os.makedirs(output_valid_folder,exist_ok=True)

    # annotation 파일 정보 불러오기
    annotation_file = [f for f in os.listdir(annotation_folder)]
    print("fine")

    # 이미지 파일과 대응되는 annoation 파일 분리
    for name in annotation_file:
        annotation_path=os.path.join(annotation_folder, name)
        
        if any(name[:-4] in filename for filename in train_set):
                copyfile(annotation_path, os.path.join(output_train_folder, name))
        elif any(name[:-4] in filename for filename in test_set):
                copyfile(annotation_path, os.path.join(output_test_folder, name))
        elif any(name[:-4] in filename for filename in valid_set):
                copyfile(annotation_path, os.path.join(output_valid_folder, name))
        else: print("cannot")

# 이미지 파일명에 따라 annotation 파일을 훈련, 테스트, 검증 세트로 분리
image_folder="C:/Users/train/images"
output_folder="C:/Users/newdata"
annotation_folder="C:/Users/judy0/train/labels"

separate_annotations_by_image_names(image_folder, annotation_folder, output_folder, train_set, test_set, valid_set)

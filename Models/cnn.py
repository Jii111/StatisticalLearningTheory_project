#라이브러리 불러오기
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
#텐서플로우 업데이트
!pip install --upgrade tensorflow

#구글 드라이브 마운트
from google.colab import drive
drive.mount('/content/drive')

# Enable eager execution
tf.config.experimental_run_functions_eagerly(True)

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

tf.config.experimental_run_functions_eagerly(True)
# 디버깅을 위한 함수 사용 예시
x = tf.constant(2.0)
tf.debugging.check_numerics(x, "x has NaN or Inf values")

#최대 픽셀 값인 255로 나눠주기
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
#이미지 셋 만들기
train_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/train_aug_cnn',  # 훈련 데이터셋 경로
    target_size=(640, 640),  # 이미지 크기 조절
    batch_size= 32,
    class_mode= 'sparse'  # 다중 클래스 분류
)


valid_generator = valid_datagen.flow_from_directory(
    '/content/drive/MyDrive/valid_cnn',  # 훈련 데이터셋 경로
    target_size=(640, 640),  # 이미지 크기 조절
    batch_size= 32,
    class_mode='sparse'  # 다중 클래스 분류인 경우
)

#클래스 수
num_classes = 5

#모델 레이어 만들기
model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(640, 640, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation='softmax')  
])

# 모델
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # 수정된 부분

#모델 확인
model.summary()
tf.debugging.set_log_device_placement(True)
tf.config.experimental_run_functions_eagerly(True)

epochs = 10
# 모델 학습 및 평가
hhistory=model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=epochs,
)

#드라이브 파일에 모델 저장하기
model.save('/content/drive/MyDrive/my_model')

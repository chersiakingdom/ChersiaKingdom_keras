
# Im# 폴더 구조를 넣어주면 데이터에 라벨링을 해준다.
import numpy as np
import tensorflow as tf
import urllib
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#1. Data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
_TRAIN_URL = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
_TEST_URL = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"
urllib.request.urlretrieve(_TRAIN_URL, 'horse-or-human.zip')
local_zip = 'horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('tmp/horse-or-human/')
zip_ref.close()
urllib.request.urlretrieve(_TEST_URL, 'testdata.zip')
local_zip = 'testdata.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('tmp/testdata/')
zip_ref.close()

TRAIN_DIR = "tmp/horse-or-human"
TEST_DIR = "tmp/testdata"

#데이터 전처리, 증강 명시
train_datagen = ImageDataGenerator(
    rescale=1./255,        # 정규화
    horizontal_flip=True,  # 수평 뒤집기
    vertical_flip=True,    # 수직 뒤집기
    width_shift_range=0.1, # 가로로 조금 움직이기
    height_shift_range=0.1,# 세로로 조금 움직이기
    rotation_range=5,      # 돌리기
    zoom_range=1.2,        # 확대
    fill_mode='nearest')

test_datagen = ImageDataGenerator(
    rescale=1./255 #정규화
)

# ImageDataGenerator 내용을 데이터에 적용
xy_train = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(300,300),
    batch_size=5,
    class_mode='binary', # y data is b 
) # 1027 images belonging to 2classes.

# validation_generator = validation_datagen.flow_from_directory(
xy_test = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(300,300),
    batch_size=5,
    class_mode='binary',
) #256 images belonging to 2 classes.

print(xy_train[0][0].shape) # xdata (5, 300, 300, 3)
print(xy_train[0][1].shape) # ydata (5,)

#2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Dropout, MaxPool2D

model = Sequential()
model.add(Conv2D(16, (3,3), 1, 'same', input_shape=(300,300,3)))
model.add(Conv2D(16, (3,3), 1, 'same'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), 2))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3,3), 1, 'same'))
model.add(Conv2D(32, (3,3), 1, 'same'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), 2))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), 1, 'same'))
model.add(Conv2D(64, (3,3), 1, 'same'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), 2))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), 1, 'same'))
model.add(Conv2D(128, (3,3), 1, 'same'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), 2))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3,3), 1, 'same'))
model.add(Conv2D(256, (3,3), 1, 'same'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), 2))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3,3), 1, 'same'))
model.add(Conv2D(512, (3,3), 1, 'same'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), 2))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(512))
model.add(Dropout(0.4))
model.add(Dense(256))
model.add(Dropout(0.4))
model.add(Dense(128))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

#3. Compile, Train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit_generator(xy_train, steps_per_epoch=206, epochs=32, verbose=2, validation_data=xy_test)
#4. Evaluate, Predict
results = model.evaluate_generator(xy_test)
print(results)
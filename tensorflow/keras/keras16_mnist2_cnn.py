import numpy as np
from tensorflow.keras.datasets import mnist

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255. #x_test 도 해줘야함!!
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.
# data = 0 ~ 255 까지의 값 전달 -> 연산 너무커짐 -> 데이터 전처리(정규화_minmax scaling)를 통해 0~1 사이로 변환해주기
# 가장 큰 값으로 나눠주면 됨. 최솟값이 0 이 아닌경우에는 minmax scalier 씀.
#x_train = x_train.reshape(60000, 14, 14, 4)

#1-2 원핫인코딩
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()# 이미지이기 때문에 Conv2D는 총 4차원 인풋해줘야함.
model.add(Conv2D(filters=30, kernel_size=(2,2), padding= 'same',
                 strides =1, input_shape = (28, 28, 1))) #데이터 갯수 제외하고(행무시)
# n, 28, 28, 30 넘어감
model.add(Conv2D(40,(2,2), activation = 'relu')) #4차원 받아들임 (데이터 갯수까지)
model.add(Conv2D(20,(2,2)))
model.add(Flatten())
model.add(Dense(100, activation = 'relu'))
model.add(Dense(10, activation = 'softmax')) #0~9까지 총 10개

#3. 컴파일, 훈련
#model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3, mode = 'min')

model.fit(x_train, y_train, epochs = 10, validation_split=0.2, verbose = 1)
# batch_size 안써주면 32개. 

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print("loss : ", results[0])
print("acc : ", results[1])
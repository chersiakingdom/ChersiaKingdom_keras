# 이미지를 Dense(2차원_행,컬럼) 모델로 만들기 :: (6만, 784) 개의 컬럼으로 인지/ 28개로 끊기.
# shape 만 맞으면 어떤 <모델>이든 구현 가능. 통상적으로 이미지 - conv 로 하지만 dnn, lstm 도 써볼 필요가 있음.

import numpy as np
from tensorflow.keras.datasets import mnist

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
x_train = x_train.reshape(60000, 28 * 28 * 1).astype('float32')/255. #x_test 도 해줘야함!!
x_test = x_test.reshape(10000, 784).astype('float32')/255.

#1-2 원핫인코딩
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Dense(50, activation = 'relu', input_shape = (28 * 28, )))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(10, activation = 'softmax')) #10가지

#3. 컴파일, 훈련
#model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3, mode = 'min')

model.fit(x_train, y_train, epochs = 10, validation_split=0.2, verbose = 1)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print("loss : ", results[0])
print("acc : ", results[1])
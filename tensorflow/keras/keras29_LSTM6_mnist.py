# 이미지 data를 cnn 모델이아닌 LSTM 모델로 풀기

import numpy as np
from tensorflow.keras.datasets import mnist

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 시계열 데이터에 넣을 것이므로 (60000, 28, 28) (10000, 28, 28) *3차원 인 mnist 데이터를 
# reshape 하지 않고 넣는다
x_train = x_train.astype('float32')/255. # 범위를 0 ~ 1로 변환하여 layer 를 거치면서 값의 폭등을 막는다. 
x_test = x_test.astype('float32')/255. 

# one hot encoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()

model.add(LSTM(256, activation='relu', input_shape=(28, 28))) #flatten 안해줘도 됨.
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(10, activation='softmax')) #10종류

# print(model.summary())

# 3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=10, validation_split=0.2, verbose=1, batch_size=64)

# 4. 평가 예측
results = model.evaluate(x_test, y_test)
print("loss : ", results[0])
print("acc : ", results[1])

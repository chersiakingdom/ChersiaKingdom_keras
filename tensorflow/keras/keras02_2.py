import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10]) 
y_train = np.array([2,4,6,8,10,12,14,16,18,20])
# weight = 2 , b = 0
x_test = np.array([101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
y_test = np.array([111, 112, 113, 114, 115, 116, 117, 118, 119, 120])
#weight = 1 , b = 10
x_predict = np.array([111, 112, 113])
#2 모델 구성
model = Sequential()
model.add(Dense(20, input_dim = 1)) # <- 아웃풋. 히든 레이어 없음
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(1))


#3. 
model.compile(loss = 'mse', optimizer= 'adam')
model.fit(x_train, y_train, epochs = 500, batch_size = 1)

#4.
loss = model.evaluate(x_test, y_train, batch_size = 1)
print('loss: ', loss)

result = model.predict([x_predict])
print('result :', result)

#과제3 : 잘 만들기 
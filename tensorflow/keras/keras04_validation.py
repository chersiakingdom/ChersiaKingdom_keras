import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x_train = np.arange(1,6)
y_train = np.arange(1,6)

x_validation = np.arange(14,17)
y_validation = np.arange(14,17)

x_test = np.arange(9,12)
y_test = np.arange(9,12)

model = Sequential()
model.add(Dense(20, input_dim = 1, activation = 'relu'))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1))
# activation 은 뭐지?
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 300, batch_size = 1, 
          validation_data=(x_validation, y_validation))
# 모의고사 풀면서 훈련 .!
loss = model.evaluate(x_test, y_test, batch_size = 1)
print('loss = ', loss)

result = model.predict([9]) #input 과 동일한 shape 여야함.
print('result = ' , result)


'''
선생님 기존의 train dataset에 14, 15, 16을 넣어서 학습하는것과 따로 validation으로 빼서 14,15,16 을 학습하는 것은 어떤 차이가 있는지 궁금합니다
x_validation, y_validation 은 안넣나요
'''

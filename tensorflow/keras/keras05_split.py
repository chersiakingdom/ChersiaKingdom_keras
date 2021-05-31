from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array

#1. 데이터
x = np.array(range(1, 101))
# x2 = array(range(1, 101))
# 같은 데이터임.
y = array(range(101,201))

#인덱스임.
x_train = x[:60] # 0번부터 59번까지 60개의 데이터 잘라오기
x_val = x[60:80] # 60부터 79까지 총 20개
x_test = x[80:] # 80부터 100 까지 총 20개

y_train = y[:60] # 0번부터 59번까지 60개의 데이터 잘라오기
y_val = y[60:80] # 60부터 79까지 총 20개
y_test = y[80:] # 80부터 100 까지 총 20개

#2
model = Sequential()
model.add(Dense(20, input_dim = 1, activation = 'relu'))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1))

#3.
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train,y_train, epochs = 300, batch_size = 1, validation_data=(x_val, y_val))

#4.
loss = model.evaluate(x_test, y_test, batch_size = 1)
print('loss = ', loss)

result = model.predict([30])
print('result = ', result)













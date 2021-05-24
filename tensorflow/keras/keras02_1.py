import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_train = np.array([1, 2, 3, 4, 5])
y_train = np.array([1, 2, 3, 4, 5])
x_test = np.array([6,7,8])
y_test = np.array([6,7,8])

# 과제0. 깃허브 만들기 keras 
# 과제1. 네이밍룰 알아오기

#2. 모델구성
model = Sequential()
model.add(Dense(20, input_dim=1))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(1))


#3. 학습
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=500, batch_size =1)


#4. 예측 및 평가
loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss =', loss)

result = model.predict([9])
print('result =', result)

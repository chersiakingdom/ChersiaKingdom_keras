# 시계열 데이터를 LSTM 이 아니라 Dense 를 이용해서 푸는 방식
# 아래 예제처럼 시계열 데이터의 양이 많지 않은 경우에는 Dense를 사용하는게 더 잘 나올 수도 있음
'''
LSTM 에서 연속된 데이터를 자르기 위해 reshape 를 쓸 수도 있으나
별도의 다른 함수를 써서 분리하기도함
'''
import numpy as np

#1.
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12], [20,30,40],
              [30,40,50], [40,50,60]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x_pred = np.array([50,60,70])

# x.shape = (13, 3)
# x_pred.shape = (3,)
x_pred = x_pred.reshape(1,3)

import numpy as np

#2.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 노드의 갯수는 대개 2의 배수로 넣음.
model = Sequential()
# model.add(LSTM(32, input_shape = (3, 1), activation= 'relu')) 
model.add(Dense(64, activation = 'relu', input_shape=(3, )))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(256, activation= 'relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation= 'relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation= 'relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation= 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

#model.summary()

#3.
model.compile(loss = 'mae', optimizer= 'adam')
model.fit(x, y, epochs = 500, batch_size=1)

#4. 
results = model.evaluate(x, y)
print('results : ', results)
      
y_pred = model.predict(x_pred)
print("y_pred :", y_pred) 

# 핵심은 어떤 데이터이든 간에 shape 만 잘 맞춰주면 DNN 을 쓰든, LSTM 을 쓰든, CNN 을 쓰든 상관없다.
# 그림을 CNN 이 아닌 시계열 처리 방식의 LSTM 으로 처리하라는 함정문제가 나오기도함
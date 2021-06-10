import numpy as np

#1.
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12], [20,30,40],
              [30,40,50], [40,50,60]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x_pred = np.array([50,60,70])

x = x.reshape(13,3,1)
x_pred = x_pred.reshape(1,3,1)
#print(x.shape)
#print(y.shape)
#print(x_pred)

#내가 원하는 것은 80 일때.

import numpy as np

#2.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

model = Sequential()
model.add(LSTM(32, input_shape = (3, 1), activation= 'relu', return_sequences=True)) 
# LSTM의 출력 ----> 2차원 데이터로 나왔다.(3차원에서 1개 적어짐)
# Conv2D나 Dense는 원래 1개씩 올려서 출력됨. . 
#ex. Conv2D input 3차원 -> 출력 4차원 -> 바로 다시 Conv2D에 넣을 수 있음.
# 다음레이어로 넘어갈때는 input 이 아니라 원하는 데이터의 차원이 기준
# LSTM을 계속 쌓을 수 있는데, 출력값이 순차적 데이터가 아니므로 나온값에 대해 다시금 정제를 해줘야함.
# reshape 포함. . 
model.add(LSTM(32, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
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
# 현재 회귀모델

#결과값 --- 79.15
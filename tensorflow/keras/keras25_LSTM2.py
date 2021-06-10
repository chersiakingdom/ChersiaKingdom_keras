import numpy as np
from tensorflow.python.keras import activations


#1.
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

x = x.reshape(4, 3, 1) 
#(batch(몇번시행), timesteps(몇개씩 잘라서 예측), feature)

#2.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

model = Sequential()
model.add(LSTM(32, input_shape = (3, 1), activation= 'relu')) 
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
      
x_pred = np.array([5,6,7]) #reshape 해줘야함! 
# (3, ) -> (1, 3, 1) 로... [[[5], [6], [7]]]
x_pred = x_pred.reshape(1,3,1)

y_pred = model.predict(x_pred)
print("y_pred :", y_pred)
# 현재 회귀모델


'''
1번 회귀문제
2번 Mnist/fashion mnist문제(conv, flatten, )
3번 CSV 다운 -> 이진분류 (말/사람분류)/ 다중분류(가위바위보)
** 함정: 이진분류->softmax로 풀라고 나옴.(다중분류로..)
4번 자연어처리(Embedding)
5번 시계열(LSTM)<-mae

링크드인 다운로드받아두기
keras.io / tensorflow.org 등의 사이트 사용하기

'''
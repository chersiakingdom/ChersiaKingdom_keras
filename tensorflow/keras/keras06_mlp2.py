import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[10, 85, 70], [90, 85, 100], [80, 50, 30], [43, 60, 100]])
# 4, 3
y = np.array([75, 65, 33])
# 4,
# x = np.transpose(x)

# 2. 모델 구성
model = Sequential()
# x = np.transpose(x)

model.add(Dense(10, input_shape=(3,), activation= 'relu')) # 열만기재, x 의 모양
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1)) # y 의 모양. y 값은 스칼라 10개짜리 1개이므로... 나중에 2, 3차원 나올수있음
# 2개 칼럼 이상 예측할때는 숫자 변할 수 있음


model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = 300, batch_size = 1)


loss = model.evaluate(x, y, batch_size = 1)
print('loss = ' , loss)

y_pred = [[60, 30, 20]]
# y_pred = np.transpose(y_pred)

y_pred = model.predict([y_pred])
print('predict = ' , y_pred)

# 함수형 모델 (N : 1)

import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#1
x = np.array([range(100), range(301, 401), range(1, 101), range(100), range(301, 401)]) # (5, 100)
y = np.array(range(711, 811)) # (1, 100)

x = np.transpose(x) # (100, 5)
y = np.transpose(y) # (100, 1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)

print(x_train.shape) # (80, 5)
print(y_train.shape) # (80, 1) 

#2
input1 = Input(shape=(5,))
dense1 = Dense(3)(input1) 
dense2 = Dense(4)(dense1)
dense3 = Dense(5)(dense2)
output1 = Dense(1)(dense3) 

model = Model(inputs=input1, outputs=output1)
# 함수형 모델은 모델 구성이 완료 된 후 시작 레이어와 끝 레이어를 명시하여 선언해야함

model.summary()

# model = Sequential()
# model.add(Dense(3, input_shape=(5,)))
# model.add(Dense(4))
# model.add(Dense(5))
# model.add(Dense(2))
# output None 행을 무시했기 때문
# 2차원이 아닌 그 이상에서도 행 정보는 None 으로 나타남
# Sequential 모델에서는 input layer 에 대한 정보x

#3
model.compile(loss='mse', optimizer='adam', metrics=['mae', 'acc'])
model.fit(x_train, y_train, batch_size=1, epochs=300, verbose=0)

#4
result = model.evaluate(x_test, y_test, batch_size=1)
print("result : ", result)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error 

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))
print("mse : ", mean_squared_error(y_test, y_predict))


from sklearn.metrics import r2_score

print('R2 : ', r2_score(y_test, y_predict)) # r2 가 높다-> 그은 직선에 인접해 있는 점들이 많다
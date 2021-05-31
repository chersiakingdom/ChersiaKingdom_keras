# 과제_ 강제로 안좋게 만들기
# R2를 음수가 아닌 0.5 이하로 만들기
#1. 레이어는 인풋과 아웃풋을 포함해 6개 이상 O
#2. batch_size - 1 O
#3. epochs = 100 이상 O
#4. 히든 레이어의 노드 갯수는 10 이상 1000 이하 O
#5. 데이터 조작 금지 O

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error

x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9,10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

x_test = np.array([11, 12, 13, 14, 15])
y_test = np.array([11, 12, 13, 14, 15])

x_pred = np.array([16, 17, 18])


def my_RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
    
#2. 
model = Sequential()
# model.add(Dense(10, input_dim = 2)) #열만 기재

model.add(Dense(10, input_dim=1, activation= 'relu')) 
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1)) 
   
#3.
model.compile(loss = 'kld', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 100)

#4. 
result = model.evaluate(x_test, y_test, batch_size = 1)
print('result = ', result)


y_predict = model.predict(x_test)
#print('mse :' , mean_squared_error(y_test, y_predict))
#print("RMSE : " , my_RMSE(y_test, y_predict))
from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict)
print('R2 : ' , R2)

# print(model.predict([x_pred]))
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9,10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

x_test = np.array([11, 12, 13, 14, 15])
y_test = np.array([11, 12, 13, 14, 15])

# RMSE , R2 ( y_test, y_pred) 넣어주면 됨.
    
#2. 
model = Sequential()
# model.add(Dense(10, input_dim = 2)) #열만 기재

model.add(Dense(10, input_dim=1, activation= 'relu')) # 열만기재, x 의 모양
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1)) # y 의 모양. y 값은 스칼라 10개짜리 1개이므로... 나중에 2, 3차원 나올수있음
# 2개 칼럼 이상 예측할때는 숫자 변할 수 있음

#3. 
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 100, batch_size = 1)

#4. 
result = model.evaluate(x_test, y_test, batch_size = 1)
print('result = ', result)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    
print('mse :' , mean_squared_error(y_test, y_predict))
print("RMSE : " , RMSE(y_test, y_predict))


from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict)
print('R2 : ' , R2)

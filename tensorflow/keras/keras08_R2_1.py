import numpy as np
from numpy.lib.function_base import _DIMENSION_NAME

x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [11,12,13,14,15,16,17,18,19,20]])
# 현재 x = (2, 10) 2행 10열
# x = (10,2) / y = (10,) 여야 모델 돌아감. 반대는 안됨.

y = np.array([1,2,3,4,5,6,7,8,9,10])
'''

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, test_size=0.2  #, shuffle=True
    ) 
'''

x = np.transpose(x) # 이렇게 상관관계를 갖고있음.
# x1 , x2 , y 한 row 끼리상관. 

# 열우선 행무시 . . input_shape=(10,2) 아님!!
# 행 삭제하고 열만 기재

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

    
    
model = Sequential()
# model.add(Dense(10, input_dim = 2)) #열만 기재

model.add(Dense(10, input_shape=(2,), activation= 'relu')) # 열만기재, x 의 모양
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(1)) # y 의 모양. y 값은 스칼라 10개짜리 1개이므로... 나중에 2, 3차원 나올수있음
# 2개 칼럼 이상 예측할때는 숫자 변할 수 있음


model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
model.fit(x, y, epochs = 100, batch_size = 1)

result = model.evaluate(x, y, batch_size = 1)
print('result = ', result)

y_predict = model.predict(x)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    
print('mse :' , mean_squared_error(y, y_predict))
print("RMSE : " , RMSE(y, y_predict))

from sklearn.metrics import r2_score
R2 = r2_score(y, y_predict)
print('R2 : ' , R2)


# x_pred = [[11, 12, 13], [21, 22, 23]]
# x_pred = np.transpose(x_pred)

# y_pred = model.predict([x_pred])
# print('predict = ' , y_pred)


# 꼭 sklearn 임포트하는거 어떻게해보자 ㅠ
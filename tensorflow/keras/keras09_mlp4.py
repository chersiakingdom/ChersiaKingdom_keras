# 다:1 mlp

import numpy as np
x = np.array([range(100), range(301, 401), range(1, 101)])

y = np.array(range(711, 811))

# print(x.shape)
# print(y.shape)

x = np.transpose(x) #100 , 3

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = True, train_size=0.8, random_state=66
    
    )
# 랜덤으로 섞일때 그냥 섞이면 모델 구성에 영향을 미칠 수 있으므로
# 섞었을때 나오는 결과 동일하게 나옴. 66 으로
#print(x_train.shape)
#print(y_train.shape)

#2. 모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


model = Sequential()
# model.add(Dense(10, input_dim = 2)) #열만 기재
# 66 계수표대로 지정되서 랜덤하게 나옴 . 모델마다 다르면 안되니까..
model.add(Dense(20, input_shape=(3, ), activation= 'relu')) 
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(1)) 
   
#3.
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 300)

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
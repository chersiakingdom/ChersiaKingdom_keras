#다:다 mlp

import numpy as np
x = np.array([range(100), range(301, 401), range(1, 101), range(100), range(301, 401)])

y = np.array([range(711, 811), range(1,101)])


#print(x.shape)
#print(y.shape)

x = np.transpose(x) #100, 5
y = np.transpose(y) #100, 2

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = True, train_size=0.8, random_state=66
    
    )
print(x_train.shape) # 80, 5
print(y_train.shape) # 80, 5


#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


model = Sequential()
model.add(Dense(20, input_shape=(5, ), activation= 'relu')) 
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(2)) 
   
#3.
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 300, batch_size=1,
          verbose=0)
# 굳이 훈련하는거 눈에 안보이게 해줌!

'''
verbose = 0 : 눈에 다 안보임
verbose = 1 : loss, metrics 다 표시됨. 디폴트
verbose = 2 : 진행되는 프로그레스 바가 없어짐. 깔끔..
verbose = 3, 4, 5... : epo만 나옴.
'''


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
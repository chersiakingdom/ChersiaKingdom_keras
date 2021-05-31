from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import tensorflow as tf
import numpy as np

#1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
x = np.arange(1,11)
y = np.array([1,2,4,3,5,5,7,9,8,11])
# 컨트롤 + / -> 주석, 주석풀기
# 잡아서 하면 몽땅 주석처리됨

#2. 모델만들기
model = Sequential()
model.add(Dense(20, input_shape= (1,))) # 아니 이건 output 이라니까
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1))



#3. 학습
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = 500, batch_size = 1)


#4. 
y_pred = model.predict(x)
print(y)
print(y_pred)


import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.plot(x, y_pred, color = 'red')
plt.savefig()
plt.show()

'''
스칼라(0차원) -> 벡터(1차원) -> 행렬(2차원) -> 그 이상의 차원 "텐서"
2       [1,2]     [[1,2]] -> 2차원 이상은 input_shape=(2,1)
x       (1, )(벡터가 1개다)=input_dim=1 @@ 벡터는 1차원
shape : 구조, 모양 ex. 1행 2열

과제
1. RMSE , MAE 정리 (+ mse)
2. R2 정리

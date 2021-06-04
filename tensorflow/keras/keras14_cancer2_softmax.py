#이진분류
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.utils import validation

#1. 데이터
data = load_breast_cancer()
x = data.data
y = data.target

#print(x.shape, y.shape)
#print(data.feature_names)
#print(data.DESCR)
#print(x[:5])
#print(y[:5])

from sklearn.model_selection import train_test_split
#x_train, y_train, x_test, y_test = train_test_split(
#    x, y, shuffle=True, train_size = 0.8, random_state = 66
#    ) 이렇게 쓰면 안됨.... . . . 
# 주의. 순서대로 써야함!!!!!!!!!!!!!!!!!!!!! 
# x 먼저 다 쓰고. 그 다음 y 쓰기.
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)

#2 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation = 'relu', input_shape=(30,)))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(2, activation = 'softmax')) 
# 노드의 갯수 잡아줘야함.

#3
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
# 머신의 판단 기준
if 
model.fit(x_train, y_train, batch_size=1, epochs=50, verbose=1, validation_split=0.2)
# binary_crosee ~~~ <- 0.5 이상이면 1, 0.5 이하면 0으로 생각함.
"""
#4.
print("loss : ", result[0])
print("accuracy : ", result[1]) 

y_predict = model.predict(x_test) 
# print("Input : ", x_test[:5])
print("TrueOutput : ", y_test[:5])
print("PredOutput : ", y_predict[:5])

'''
## 원핫인코딩 OnehotEncoding
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
'''
"""
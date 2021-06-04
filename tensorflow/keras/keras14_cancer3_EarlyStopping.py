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


from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, mode = 'min')
# auto 로 쓰면 자동으로 해줌.


model.fit(x_train, y_train, batch_size=1, epochs=50, verbose=2, validation_split=0.2,
          callbacks=[early_stopping])
# 반환해준다. 불러서 쓰겠다.
# binary_crosee ~~~ <- 0.5 이상이면 1, 0.5 이하면 0으로 생각함.


#4.
result = model.evaluate(x_test, y_test, batch_size = 1)
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

지금까지 한 것  
DNN : deep 뉴럴 네트워크
CNN : converlution 뉴럴 네트워크

'''


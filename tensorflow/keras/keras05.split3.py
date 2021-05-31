# validation_split 사용

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array

#1. 데이터
x = np.array(range(1, 101))
# x2 = array(range(1, 101))
# 같은 데이터임.
y = array(range(101,201))

#인덱스임.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, test_size=0.2  #, shuffle=True
    ) 
# x_test, x_val, y_test, y_val = train_test_split(
#     x_test, y_test, train_size = 0.5)

#shuffle false 로 해주면 순서대로 나옴. . 
# 디폴트값은 True(랜덤)
#print(x_test.shape)
#print(x_train.shape)

# 캐글 50~100, 데이톤 문제 ~20, 해커톤 등... 문제 많이 풀어보자.

'''
x_train = x[:60] # 0번부터 59번까지 60개의 데이터 잘라오기
x_val = x[60:80] # 60부터 79까지 총 20개
x_test = x[80:] # 80부터 100 까지 총 20개

y_train = y[:60] # 0번부터 59번까지 60개의 데이터 잘라오기
y_val = y[60:80] # 60부터 79까지 총 20개
y_test = y[80:] # 80부터 100 까지 총 20개
'''
#2
model = Sequential()
model.add(Dense(20, input_dim = 1, activation = 'relu'))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1))

#3.
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 300, batch_size = 1, 
          validation_split = 0.2)

#4.
loss = model.evaluate(x_test, y_test, batch_size = 1)
print('loss = ', loss)

result = model.predict([101, 102, 103])
print('result = ', result)












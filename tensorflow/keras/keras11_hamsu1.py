# Sequential model -> 함수형 모델로 바꾸기

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
from tensorflow.keras.models import Sequential, Model
# 함수형 과 순차형으로 나뉜다. 나머지는 이 둘을 조합하는 것..
# Sequential 은 다양한 모델 구성이나 다양한 데이터를 받기 힘들다.
# 함수형 모델의 예 : "앙상블 모델": 
# 각 각 만들어놓은 모델의 아웃풋 값을, 다시 인풋 값으로 받아 (dim - 2)
# 다시 한 모델로 만들어 예측할 수 있다. 
from tensorflow.keras.layers import Dense, Input
# 함수형 모델은 import 할 때 부터 input 을 명시해주어야 한다! 

# model = Sequential()
# model.add(Dense(3, input_shape=(5, ), activation= 'relu')) 
# model.add(Dense(4))
# model.add(Dense(2))
# model.summary()
# 연산의 갯수 + bias(아웃풋 갯수만큼 있음) = 파라미터(w, b)의 갯수
# 한 y당 하나의 b .... x 갯수만큼 있는거 아님. 
# 5 -> 3 -> 4 -> 2 ( 총 4 층 모델)


input1 = Input(shape=(5, )) # 함수형 모델. 순차적 모델의 첫 레이어와 동일
dense1 = Dense(3)(input1)
dense2 = Dense(4)(dense1)
output1 = Dense(2)(dense2) #뜬금없이 다른앨 받아올수도 있는건가? 다양한 변화가 가능하겠네
model = Model(inputs = input1, outputs = output1)
model.summary()
# 파라미터 같다(연산 횟수) -> 기본적인 성능 같다
# 함수형은 쌓아놓고 모델을 선언한다. 최초 인풋이름, 마지막 출력 아웃푹 이름.


#3.
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 300, batch_size=1,
          verbose=0)
# 굳이 훈련하는거 눈에 안보이게 해줌!

"""
verbose = 0 : 눈에 다 안보임
verbose = 1 : loss, metrics 다 표시됨. 디폴트
verbose = 2 : 진행되는 프로그레스 바가 없어짐. 깔끔..
verbose = 3, 4, 5... : epo만 나옴.
"""

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

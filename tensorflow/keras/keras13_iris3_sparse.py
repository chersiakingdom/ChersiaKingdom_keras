#분류모델
import numpy as np

from sklearn.datasets import load_iris
# 꽃 데이터를 통해서 어떤 꽃인지 분류 ***회귀X, 분류O
# 3개의 분류 : 0, 1, 2 

#1
data = load_iris()

x = data.data
y = data.target 

#print(x.shape, y.shape) # (150, 4) (150, )

## 원핫인코딩 OnehotEncoding (sparse의 경우 안해도됨)
#from tensorflow.keras.utils import to_categorical
#y = to_categorical(y)

#print(data.feature_names)
#print(data.DESCR) # Description

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)

#2
input1 = Input(shape=(4,))
dense1 = Dense(30, activation = 'relu')(input1) 
dense2 = Dense(40)(dense1)
dense21 = Dense(40)(dense2)
dense3 = Dense(30)(dense21) 
output1 = Dense(3, activation='softmax')(dense3)  # output에 분류 att 갯수 넣기 (출력되는 타겟 종류)
model = Model(inputs=input1, outputs=output1)
# 분류 모델에서도 loss 지표는 그대로 사용. y = wx + b 직선 식을 그대로 이용하기 때문
# 다만 평가가 달라야 분류라는것을 알수 있으므로, metrics와 model.evaluate 가 변경되어야 함
# R2 는 필요가 없으니 지움
# 엑티베이션 명시 안해주면 디폴트로 linear 

#3
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
# 원핫을 알아서 자동으로 해주고 한다 .  
#model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=1, epochs=100, verbose=2, validation_split=0.1)
# 분류 loss 종류로는 sparse cate~ / categorcal_crossentropy 씀.
# categorical-clossentropy가 0.xxxxx 를 0, 1, 2 로 만들어준다.

# 4
result = model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", result[0])
print("acc : ", result[1]) 
 
# evaluate 할때 결과물은 항상 loss 와 metrics 가 쌍으로 나온다

y_predict = model.predict(x_test) 
print("Input : ", x_test[:5])
print("TrueOutput : ", y_test[:5])
print("PredOutput : ", y_predict[:5])

# 분류 종류 :: 다중(여러가지), 이산, ... 

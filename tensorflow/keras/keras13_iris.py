import numpy as np

from sklearn.datasets import load_iris
# 꽃 데이터를 통해서 어떤 꽃인지 분류 ***회귀X, 분류O
# 3개의 분류 : 0, 1, 2 

#1
data = load_iris()

x = data.data
y = data.target 

#print(x.shape, y.shape) # (150, 4) (150, )

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
output1 = Dense(3, activation='softmax')(dense3) 
# output에 분류 att 갯수 넣기

model = Model(inputs=input1, outputs=output1)
# 분류 모델에서도 loss 지표는 그대로 사용. y = wx + b 직선 식을 그대로 이용하기 때문
# 다만 평가가 달라야 분류라는것을 알수 있으므로, metrics와 model.evaluate 가 변경되어야 함
# R2 는 필요가 없으니 지움

#3
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=1, epochs=100, verbose=5, validation_split=0.1)

# 4
result = model.evaluate(x_test, y_test, batch_size=1)
print("result : ", result) 
# evaluate 할때 결과물은 항상 loss 와 metrics 가 쌍으로 나온다

y_predict = model.predict(x_test) 
print(y_test[:5])
print(y_predict[:5])

'''
    분류 모델은 소수점 조금이라도 다르면 다른 것으로 인식하기 때문에
    소수점을 반올림해서 동일한 대상으로 바꿔줘야할 필요가 있음
    이때 쓰이는게 활성화 함수 activation
    activation 은 값의 폭발을 막기위한 용도로 사용됨
    만약 y = wx + b ~ 에서 w 의 값이 너무 크면
    출력이 너무 크게 나와서 예측하려는 값과의 차이점이 너무 크게 됨
    따라서 w 의 값을 한정시키기 위한 용도로 쓰이는게 activation 이다
    (보통 0 ~ 1 사이의 소수로 지정함)
    가장 대표적인 activation 이 relu 이며
    Dense 인스턴스를 만들때 옵션으로 넣으면 된다
    다중 분류
    : 분류해야하는게 3개 이상일때 
    -> 다중 분류에서 출력 노드 activation 은 무조건 softmax 씀
    * activation 의 기본 값은 linear
    relu 는 0 ~ 1 사이의 값으로 지정하기 위해서 쓰이는 것
    loss 기준으로 쓰이는 것은 sparse_categorical_crossentropy 이다
'''

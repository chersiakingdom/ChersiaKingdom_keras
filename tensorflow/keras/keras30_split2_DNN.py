# 연속된 값을 갖는 리스트를 시계열 데이터 분석(LSTM)에 넣기 위해서 
# split_x 함수로 분리시키고, 모델을 구성하고 컴파일 및 훈련 평가 예측도 해본다

import numpy as np

#1. Data
size = 6
data = np.array(range(1,11))
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    aaa = np.array(aaa)
    x = np.array(aaa[:, :-1]) # 이렇게 해두면 편함. 
    y = np.array(aaa[:, -1])
    return x, y
x_data, y_data = split_x(data, size)

#2. Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

input1 = Input(shape=(x_data.shape[1:]))
dense1 = Dense(64, activation='relu')(input1)
dense2 = Dense(32, activation='relu')(dense1)
dense3 = Dense(16, activation='relu')(dense2)
output1 = Dense(1)(dense3)

model = Model(inputs=input1, outputs=output1)
#model.summary()

#3. Compile, Train
model.compile(optimizer='adam', loss='mse')
model.fit(x_data, y_data, epochs=126, batch_size=1)

#4. Evaluate, Predict
result = model.evaluate(x_data, y_data, batch_size=1)
#x_pred = np.array([[6, 7, 8, 9, 10]])
y_pred = model.predict([[6,7,8,9,10]])
print("loss : ", result)
print("pred : ", y_pred)

y_predict = model.predict(x_data)
print(x_data.shape)
print(y_data.shape)

from sklearn.metrics import r2_score
R2 = r2_score(y_data, y_predict)
print("R2 : ", R2)
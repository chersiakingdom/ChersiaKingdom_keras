import numpy as np

size = 5

a = np.array(range(1, 11))


def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i + size)]
        aaa.append(subset)
    # print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print(dataset)

x = dataset[:, :4] # 행은 전체 다 가져오고, 열은 4번째 열 '이전'(0 ~ 3)까지만 가져오겠다
y = dataset[:, 4] # 행은 전체 다 가져오고, 열은 마지막 열(4번째열)만 가져오겠다
# print(x)
# print(y)

x_pred = np.array([5, 6, 7, 8])

print(x.shape)
print(y.shape)

x = x.reshape(6, 4, 1)

# 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(32, input_shape=(4, 1)))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))

# 컴파일 및 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=50, batch_size = 1)

# 평가 및 예측
results = model.evaluate(x, y, batch_size = 1)
print("results : ", results)

x_pred = x_pred.reshape(1, 4, 1)

y_pred = model.predict(x_pred)
print("y_pred : ", y_pred)

y_predict = model.predict(x)
from sklearn.metrics import r2_score
R2 = r2_score(y, y_predict) #실제값 , 예측값
print('R2 : ' , R2)


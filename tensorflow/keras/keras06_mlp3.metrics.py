import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[10, 85, 70], [90, 85, 100], [80, 50, 30], [43, 60, 100]])
# 4, 3
y = np.array([75, 65, 33, 50])
# 4,


# 2. 모델 구성
model = Sequential()
# x = np.transpose(x)

model.add(Dense(10, input_shape=(3,), activation= 'relu')) # 열만기재, x 의 모양
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1)) # y 의 모양. y 값은 스칼라 10개짜리 1개이므로... 나중에 2, 3차원 나올수있음
# 2개 칼럼 이상 예측할때는 숫자 변할 수 있음

#3. 
#model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
#model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
#model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc']) # accuracy
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae','acc'])
# ([ ]) 이렇게 리스트로 받으면 인수를 2개 이상 받을 수 있다고 한번 생각을 해보자


#회귀, 분류 에 넣어줄 수 있는 평가지표가 다르다.
# 분류 <- accuracy, precision, Recall, F1 score, fall out .. 
# 회귀 <- mse, mae, rmse, mape, R2 ...

# R2가 0.7 정도 되면 70% 정도 맞았구나 생각함. (확실하게는 아님)

# metrics 에 넣으면 학습할때 지표에 대한걸 보여줌
model.fit(x, y, epochs = 300, batch_size = 1)


loss = model.evaluate(x, y, batch_size = 1)
print('loss = ' , loss)

y_pred = [[60, 30, 40]]

y_pred = model.predict([y_pred])
print('predict = ' , y_pred)



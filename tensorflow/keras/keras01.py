import numpy as np
import tensorflow as tf

#1. 데이터 준비
x = np.array([1,2,3])
y = np.array([1,2,3])
# 특기 : 데이터 전처리


#2 모델구성 ( 함수형 , sequential )
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(30, input_dim =1)) # 모델에 dense 레이어 붙이겠다
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1))

#3 컴파일, 훈련 (로스 최적화)
model.compile(loss='mse', optimizer = 'adam') #지표mse : 더 산포 안되게함. 제곱의 오차
model.fit(x, y, epochs =100, batch_size = 1) 
# 취미 : 하이퍼파라미터튜닝(2, 3번단계에서 모델 조절.. 예측값 튜닝)


#4 예측, 평가
loss = model.evaluate(x,y, batch_size =1)
print('loss:', loss)

results = model.predict([4])
print('result :' , results) #정확도

#과제
#mse는 무엇인가
# batch_size 디폴트값은 무엇인가 -- 다 넣기
# 하이퍼피라이터 튠을 해볼것

# kingkeras@naver.com

# loss 랑 result 는 무슨 관곈데??



''' 2주차 숙제!
깃허브 만들기!
repository : 아이디_keras 
1. 메모리 큰거
2. cuda 코어 많은거








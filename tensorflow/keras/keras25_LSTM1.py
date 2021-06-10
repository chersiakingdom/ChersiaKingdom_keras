# 이미지, 시계열 이 주요 토픽이 됨.
# 시계열 데이터 -- 주가, 날씨, 기상, 전력량, 미세먼지 등

#LSTM 순서가 있는 data에서 잘 먹힌다.
#RNN(순환신경망)의 대표적인 알고리즘이 LSTM임.

import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

x = x.reshape(4, 3, 1) #1은.. 연산 갯수 추가해주기(몇개씩 작업할건지)
# print(x.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, input_shape = (3, 1))) #4행 3열을 reshape
model.add(Dense(10))
model.add(Dense(1))

#Dense는 2차원 / input(N, 컬럼)은 1차원, 
#Conv2D는 4차원 / input(N, 가로, 세로, 칼라])은 3차원,
#RNN계열(LSTM)은 3차원 / input(N, 컬럼, 연산갯수)은 2차원
# 연산갯수 <==== * 몇개씩 잘라서 계산하는지

model.summary()

# 과제 . . . 파라미터가 왜 480, 110, 11 인지 알아보기.
# 최신거 말고(넘어렵), 타이타닉 부터 하나하나씩 풀기 
# 캐글, 데이콘 문제 많이 풀어두기

'''
1번 회귀문제
2번 Mnist/fashion mnist문제(conv, flatten, )
3번 CSV 다운 -> 이진분류 (말/사람분류)/ 다중분류(가위바위보)
** 함정: 이진분류->softmax로 풀라고 나옴.(다중분류로..)
4번 자연어처리(Embedding)
5번 시계열(LSTM)<-mae

링크드인 다운로드받아두기

'''
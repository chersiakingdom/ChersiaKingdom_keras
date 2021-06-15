# 이미지 데이터는 기본적으로 그림갯수, 가로, 세로, 색상의 4차원으로 구성됨.
# 이 데이터를 Flatten 하면 그림갯수, 스칼라, 색상의 3차원으로 됨.
# 아니래.... flatten 하면 무조건 2차원이 되는거라는데?

# Conv1D는 (??Conv2D를 Flatten 한 것??), 즉 3차원임.
# 이는 같은 3차원을 요구하는 LSTM 모델과 연계시킬 수 있음.
# (다만, LSTM은 아웃풋이 한차원 낮아져서 나옴. 그래서 한번 더 써줄때는 return_sequences 해줘야했음)
# Conv1D 장점 : LSTM에 비해 연산횟수 적음, 더 빠름 
# 보통 LSTM이 너무 늦게 나오면 Conv1D를 씀

# 문제에서 Fashion MNIST, MNIST 등의 3차원(흑백) 이미지 데이터를 reshape 하지 말고 처리하라고 하면
# LSTM 보다는 Conv1D를 쓸 것.!!

import numpy as np

#1. Data
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255


from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D

model = Sequential()
model.add(Conv1D(
        filters=32, kernel_size=(2,), padding='same',
        strides=1, input_shape=(28,28))) 
#data가 가로,세로가 아니라 널려있기 때문에, conv2D와는 다르게 kernel 사이즈가 (2,) 여야함. 
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. Compile, Train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, epochs=32, verbose=2, validation_split=0.2)

#4. Evaluate, Predict
loss = model.evaluate(x_test,y_test)
print('loss :', loss[0])
print('acc  :', loss[1])

#### 일단 여기까지 ##### 
# ImageData ~ fit 까지 다시 공부해야함.



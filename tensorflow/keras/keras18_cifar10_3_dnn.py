import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.constraints import max_norm
from keraspp.skeras import plot_loss, plot_acc

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape) # 5만, 32, 32, 3 / 컬러데이터
print(y_train.shape) # 5만, 1
#1-2. dnn 모델로 reshape하기
x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

#1-3. sparse 안넣고 해보기 위해 one-hot 인코딩
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_class = y_test.shape[1]

#2. 모델
model = Sequential()
model.add(Dense(64, input_shape = (32*32*3, ), activation = 'relu', kernel_constraint=max_norm(3)))
model.add(Dropout(0.1))
model.add(Dense(128, activation= 'relu', kernel_constraint=max_norm(3)))
model.add(Dropout(0.15))
model.add(Dense(256, activation= 'relu', kernel_constraint=max_norm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation= 'relu', kernel_constraint=max_norm(3)))
model.add(Dropout(0.25))
model.add(Dense(1024, activation= 'relu', kernel_constraint=max_norm(3)))
model.add(Dropout(0.3))
model.add(Dense(512, activation= 'relu', kernel_constraint=max_norm(3)))
model.add(Dropout(0.25))
model.add(Dense(1024, activation= 'relu', kernel_constraint=max_norm(3)))
model.add(Dropout(0.3))
model.add(Dense(num_class, activation= 'softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor= 'val_loss', patience= 3, mode = 'min')

#history = model.fit(x_train, y_train, epochs = 50, verbose = 1, validation_split= 0.2 )
model.fit(x_train, y_train, epochs = 50, verbose = 1, validation_split= 0.2 )

#4. 예측 및 평가
results = model.evaluate(x_test, y_test)
print("loss : ", results[0])
print("acc : ", results[1])

'''
plot_acc(history, '(a) Accuracy')
plt.show()
plot_loss(history, '(b) loss')
plt.show()
'''


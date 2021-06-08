from matplotlib import pyplot
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.constraints import max_norm

seed = 7
np.random.seed(seed)

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape) # 5만, 32, 32, 3 / 컬러데이터
print(y_train.shape) # 5만, 1

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_class = y_test.shape[1]

#2. 모델
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1, input_shape=(32, 32, 3), 
                 padding='same', activation= 'relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), activation= 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3), activation='relu', padding = 'same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, (3,3), activation='relu', padding = 'same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3,3), activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation= 'relu', kernel_constraint=max_norm(3)))
model.add(Dropout(0.3))
model.add(Dense(512, activation= 'relu', kernel_constraint=max_norm(3)))
model.add(Dropout(0.3))
model.add(Dense(1024, activation= 'relu', kernel_constraint=max_norm(3)))
model.add(Dropout(0.3))
model.add(Dense(num_class, activation= 'softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor= 'val_loss', patience= 3, mode = 'min')

model.fit(x_train, y_train, epochs = 10, verbose = 1, validation_data= (x_test, y_test))

#4. 예측 및 평가
results = model.evaluate(x_test, y_test)
print("loss : ", results[0])
print("acc : ", results[1])

#from keras.models import load_model
#model.save('save.h1')
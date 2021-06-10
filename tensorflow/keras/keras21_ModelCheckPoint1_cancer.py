import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout

#이진분류_earlystop
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.utils import validation

#1. 데이터
data = load_breast_cancer()
x = data.data
y = data.target

#print(x.shape, y.shape)
#print(data.feature_names)
#print(data.DESCR)
#print(x[:5])
#print(y[:5])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)

#2 모델

model = Sequential()
model.add(Dense(10, activation = 'relu', input_shape=(30,)))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(2, activation = 'softmax')) 

#3
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience = 5, mode = 'min')
# auto 로 쓰면 자동으로 해줌.
modelpath = './keras/CheckPoint/k21_cancer_{epoch:02d}-{val_loss:.4f}.hdf5'
# 경로 / 세이브파일 이름 {epoch값 2자리로 들어옴 }, {소수 4자리수 loss값 들어옴 }
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', 
                     save_best_only=True, mode = 'auto') 

hist = model.fit(x_train, y_train, batch_size=1, epochs=50, verbose=2, validation_split=0.2, 
                 callbacks=[es, cp])


import matplotlib.pyplot as plt

#4.
result = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", result[0])
print("accuracy : ", result[1]) 

#y_predict = model.predict(x_test) 
# print("Input : ", x_test[:5])
#print("TrueOutput : ", y_test[:5])
#print("PredOutput : ", y_predict[:5])

print(hist)
print(hist.history.keys()) #딕셔너리키 반환(loss, acc, val_loss, val_acc)
#print(hist.history(['loss']))
#print(hist.history(['val_loss']))

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train acc', 'val_acc']) #첫번째 plot의 이름, 두번째 plot의 이름
plt.show()


'''
learning rate : 학습률---값 계속 갱신 , * optimizer 에서 나옴
최적의 loss 를 위해 최적의 weight 찾기. 적당히 설정해야함..

좋은 model 이 나오면 그 지점을 저장(weight)해야지. 그 값 저장하면 이후에 다시 훈련 안시켜도 되니까 . . .

'''


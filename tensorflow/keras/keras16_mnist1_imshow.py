import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape)

print(y_train.shape, y_test.shape)

# x.shape =  (7만, 28, 28, 1) 이때 ',1' 은 같으므로 생략되기도함.*흑백데이터 -> reshape 해줘야함.
# flatten 일때 연산 수만 맞춰주게끔 reshape 하는것 가능. 예를들어 (7만, 14, 14, 4 ) 가능
# y.shape =  (7만 , )

print(x_train[0]) # 28, 28
print(y_train[0]) # 1

#시각화
import matplotlib.pyplot as plt

plt.imshow(x_train[0], 'gray')
plt.show()
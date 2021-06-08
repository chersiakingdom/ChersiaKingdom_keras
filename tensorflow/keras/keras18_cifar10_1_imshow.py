# cifar10_ 6만, 32, 32, 3 (컬러데이터)
import numpy as np
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, x_test.shape) # 5만, 32, 32, 3 / 1만, 32, 32, 3
print(y_train.shape, y_test.shape) # 5만, 1/ 1만, 1

print(x_train[10]) 
print(y_train[10]) 

#시각화
import matplotlib.pyplot as plt
plt.imshow(x_train[1])
plt.show()

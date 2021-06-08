# fashion mnist _ 7만, 28, 28 / y = 10종류
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, x_test.shape) 
print(y_train.shape, y_test.shape)

print(x_train[10]) # 28, 28
print(y_train[10]) # 1

#시각화
import matplotlib.pyplot as plt

plt.imshow(x_train[10], 'gray')
plt.show()
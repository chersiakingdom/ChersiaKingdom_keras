# 100종류

from tensorflow.keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape) #5만, 32, 32, 3 / 5만, 1
print(x_test.shape, y_test.shape) # 1만, 32, 32, 3 / 1만, 1

i = 32

print(x_train[i])
print(y_train[i])
import matplotlib.pyplot as plt
plt.imshow(x_train[i])
plt.show()









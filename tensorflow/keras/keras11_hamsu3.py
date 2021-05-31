import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#1
x = np.array(range(711, 811)) # (1, 100)
y = np.array([range(100), range(711, 811)]) # (2, 100)

x = np.transpose(x) # (100, 1)
y = np.transpose(y) # (100, 2)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)

print(x_train.shape) # (80, 1)
print(y_train.shape) # (80, 2) 

#2
input1 = Input(shape=(1,))
dense1 = Dense(30)(input1)
dense2 = Dense(30)(dense1)  
dense3 = Dense(50)(dense2) 
dense4 = Dense(40)(dense3)
dense5 = Dense(80)(dense4)
output1 = Dense(2)(dense5)

model = Model(inputs=input1, outputs=output1)


model.summary()

# model = Sequential()
# model.add(Dense(3, input_shape=(5,)))
# model.add(Dense(4))
# model.add(Dense(5))
# model.add(Dense(2))
# model.summary() 

#3
model.compile(loss='mse', optimizer='adam', metrics=['mae', 'acc'])
model.fit(x_train, y_train, batch_size=1, epochs=300, verbose=5)

#4
result = model.evaluate(x_test, y_test, batch_size=1)
print("result : ", result)


y_predict = model.predict(x_test) 


from sklearn.metrics import mean_squared_error 

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))
print("mse : ", mean_squared_error(y_test, y_predict))


from sklearn.metrics import r2_score
print('R2 : ', r2_score(y_test, y_predict))
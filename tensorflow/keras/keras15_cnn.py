from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(filters=10, kernel_size = (2,2), strides = 1, input_shape = (5, 5, 1)))
# 연산하면 (5, 5, 1) 에서 (4, 4, 10 ) 로 됨. 
# padding = 'same'으로 하면 shape 그대로 가져감. 장점 : 손실방지
model.add(Conv2D(5, (2,2), padding = 'same')) 
#연산하면 4, 4, 10 에서-> 4, 4, 5 가 됨. 4, 4 그대로 가져가고.. 뒤에 filter값 

# 젤 앞에있는 수치 필터(노드갯수_아웃풋)로 인식
# 다음수치 커널사이즈, stride는 디폴트1
model.add(Flatten())

model.add(Dense(1))


model.summary()

# 레이어에서 이미지를 머신이 알 수 있게 수치화 하는 작업 시행
# 특성값을 찾기 위해 조각내기. 커널사이즈 = 자르는 조각 크기 . . 
# stride = 몇번씩 이동할건지, filter = (아웃풋)노드 갯수

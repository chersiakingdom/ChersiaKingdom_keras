#이미지. dnn(dense) 모델 (input = (n,)과는 달리 다차원의 연산 가능.(input = (n, n, n))
#이미지 = 숫자.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(filters=10, kernel_size = (2,2), strides = 1, input_shape = (5, 5, 1)))
# 연산하면 (5, 5, 1) 에서 (4, 4, 10 ) 로 됨. 
# padding = 'same'으로 하면 shape 그대로 가져감. 장점 : 손실방지
# shape = 갯수(행.. 무시.. none), 가로, 세로, 칼라
model.add(Conv2D(5, (2,2), padding = 'same')) 
#연산하면 4, 4, 10 에서-> 4, 4, 5 가 됨. 4, 4 그대로 가져가고.. 뒤에 filter값 
# 젤 앞에있는 수치 필터(노드갯수_아웃풋)로 인식
# 다음수치 커널사이즈, stride는 디폴트1
model.add(Flatten()) # 쫙 펼쳐줌. Dense에 넣어주기 위해 .  
model.add(Dense(10, activation = 'softmax')) 

'''
문제. 가위바위보
dense = 3, softmax , (sparse) categorical_cross_entropy

문제. 말, 사람 구분
dense = 1, sigmoid, binary_cross_entropy
~> 소프트맥스로 풀면?
dense = 2, softmax , (sparse) categorical_cross_entropy

'''
model.summary()

# 레이어에서 이미지를 머신이 알 수 있게 수치화 하는 작업 시행
# 특성값을 찾기 위해 조각내기. 커널사이즈 = 자르는 조각 크기 . . 
# stride = 몇번씩 이동할건지, filter = (아웃풋)노드 갯수

# 픽셀 25개는 특성 25개가 됨. 그래서 CNN문제를 DNN 문제로 풀기도 함.
# 즉, (5, 5, 1) -> (None, 25)
# Dense는 상황에 맞게 바꿔주기
# 예를들어, 남자인지 여자인지 맞춘다면 Dense(2, activation = 'sigmoid'. 배운것과 동일하게 하면됨)

'''
Q. 파라미터 값이 왜 50, 205 로 나오는가?
A. 
커널사이즈 * 입력이미지 채널수 * 전체(출력) 커널 갯수 + bias = 파라미터값
모든 커널에는 커널사이즈 *입력이미지 채널수 만큼의 파라미터가 있고, 
그런 커널들이 전체 커널 갯수만큼 있기 때문이다.
따라서, 첫번째 경우에서는
(2, 2) 필터 한개에는 4개의 파라미터가 있고, 1 채널에 각각 다른 파라미터들이 입력되므로 1이 곱해진다
출력 채널인 필터(10) 이므로 10을 곱해주고 더하면,
2 * 2 * 1  * 10 + 10 = 50
두번째 경우에서는
2 * 2 * 10 * 5 + 5 = 205 가 된다.


'''
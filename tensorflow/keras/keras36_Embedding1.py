from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화에요', '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요',
        '글쎄요', '별로에요', '생각보다 지루해요', '연기가 어색해요', '재미없어요', '너무 재미없다.', '참 재밌네요', 
        '배우가 잘 생기긴 했어요']

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
x = token.texts_to_sequences(docs)
print(x)

'''
{'참': 1, '너무': 2, '잘': 3, '재밌어요': 4, '최고에요': 5, '만든': 
6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 
11, '번': 12, '더': 13, '보고': 14, '싶네요글쎄요': 15, '별로에요': 
16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, ' 
재미없어요': 21, '재미없다': 22, '재밌네요': 23, '배우가': 24, '생기
긴': 25, '했어요': 26}
[[2, 4], [1, 5], [1, 3, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], 
[16], [17, 18], [19, 20], [21], [2, 22], [1, 23], [24, 3, 25, 26]]  

문제 발생 : 13, ? << 어절마다 shape가 다 다름!! ~> 모델에 넣으려면 크기를 맞춰줘야함.
가장 큰 애(단어 많은애)로 맞춤. 빈자리는 0으로 채워주기

예를 들어, [2, 4] -> [0,0,0,2,4]
[1,5] -> [0,0,0,1,5]

시계열 데이터 특징 : 모델링시 데이터의 끝부분이 많이 반영됨. (ex, 날씨데이터는 어제 날씨가 더 중요)
그렇기 때문에 2,4,0,0,0 이 아니라, 0,0,0,2,4 로 맞추는 것.

'''

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen = 5)  # <- train 할 데이터
#앞에서부터 0을 채우겠다. / maxlen 에는 가장 단어 많은 어절 넣어줌.
# post <- 뒤에부터 채우겠다.#그러면 5, 4 로 맞춰짐.

print(pad_x)
print(pad_x.shape) #13, 5 <train_shape> 여기까지 만들어주면 됨!

print(np.unique(pad_x)) #전체 중 유니크한 데이터 찍힘 ( 0 ~ 27) 
print(len(np.unique(pad_x))) #전체 갯수 찍힘 #28(단어 수)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D #학습하며 데이터 처리해줌.

model = Sequential()
model.add(Embedding(input_dim=280, output_dim=2, input_length=5)) 

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 5, 2)              56
_________________________________________________________________
lstm (LSTM)                  (None, 32)                4480
_________________________________________________________________
dense (Dense)                (None, 1)                 33
=================================================================
Total params: 4,569
Trainable params: 4,569
Non-trainable params: 0
_________________________________________________________________
'''
#model.add(Embedding(28, 2)) 
#이렇게 위와 같음. 됨. input_dim 과 output_dim 만 써주기.
# input_length 안써줘도 알아서 해줌.

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, None, 2)           56
_________________________________________________________________
lstm (LSTM)                  (None, 32)                4480
_________________________________________________________________
dense (Dense)                (None, 1)                 33
=================================================================
Total params: 4,569
Trainable params: 4,569
Non-trainable params: 0
_________________________________________________________________

'''
# 쓰는 순서반드시맞춰줘야함! 
# (None, 5 ) 받아들여서 -> (Node, 5, 7(output_dim)) 으로 들어감
# 즉, Embedding 은 2차원을 받아들여서 3차원으로 뱉어냄. LSTM 바로 넣을 수 있음.
# input_dim : 사전(단어)의 갯수_28 
  #단어사전의 갯수보다 크게 써주는건 상관 없음. 작게쓰면 문제.
# output_dim : output "노드" 갯수 (분석자 마음대로 정함)_7
  # Embedding 은 특이하게 처음에 안넣고 두번째에넣음. 
# input_lenth : trainset의 inputshape(13, 5) 에서 열을 따옴_5
  # lenth 길이는 알아서 조정해줘서 크게쓰든 작게쓰든 똑같음.
model.add(LSTM(32)) # 3차원으로 받음
model.add(Dense(1, activation='sigmoid')) #이진분류 : 긍정(1)과 부정(0)을 맞추는모델

# Embedding이 알아서 벡터화(수치화)해줌 (원핫따로할필요없어짐)
# 뒤에 000000 들어있던게 제거되며 간결해짐.

model.summary()
# 첫 연산 파라미터 갯수 : 사전 수(input_dim:28) * 아웃풋 갯수(7)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(pad_x, labels, epochs=100)

acc = model.evaluate(pad_x, labels)[1] #test 모델이 없어서 그냥 이렇게함.
print(acc)





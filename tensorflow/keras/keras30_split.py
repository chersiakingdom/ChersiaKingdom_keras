# 연속된 값을 갖는 리스트를 시계열 데이터 분석(LSTM)에 넣기 위해서 
# split_x 라는 함수를 통해서 분리시킴

import numpy as np

size = 5

a = np.array(range(1, 11))

# 쪼갤dataset, 몇개의 data씩 쪼갤지
def split_x(seq, size): # sequance를 주소를 1씩 증가시키며 size만큼의 데이터로 나눈다. 
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i + size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print(dataset)

x = dataset[:, :4] # 행은 전체 다 가져오고, 열은 4번째 열'이전'까지( 0 ~ 3열)만 가져오겠다
y = dataset[:, 4] # 행은 전체 다 가져오고, 열은 4번째 열의 값만 가져오겠다

print(x)
print(y)

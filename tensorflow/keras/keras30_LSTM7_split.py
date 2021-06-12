# 연속된 값을 갖는 리스트를 시계열 데이터 분석(LSTM)에 넣기 위해서 
# split_x 라는 함수를 통해서 분리시킴

import numpy as np

size = 5

a = np.array(range(1, 11))

# 쪼갤dataset, 몇개의 data씩 쪼갤지
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i + size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print(dataset)

x = dataset[:, :4] # 행은 전체 다 가져오고, 열은 4번째 열 (0 ~ 3)까지만 가져오겠다
y = dataset[:, 4] # 행은 전체 다 가져오고, 열은 4번째 열의 값만 가져오겠다
# 456789 여야하는데, 왜 5,6,7,8,9,10 으로 나오지?

print(x)
print(y)
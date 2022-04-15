from enum import Enum

import numpy as np

aa = np.array([[1, 2, 3], [4, 5, 6]])

a = aa[1, 2]
print(a)


class SoybeanPart(Enum):
    FirstBean = 0
    SecondBean = 1
    ThirdBean = 2
    FourthBean = 3
    FifthBean = 4
    Background = 5


print(SoybeanPart.Background.value)

v = [
    [1, 2],
     [3, 4],
     [5, 6]
     ]
res = np.asarray(v)
print(res)

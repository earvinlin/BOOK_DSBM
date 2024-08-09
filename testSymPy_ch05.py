#-- Sample 5-1 --#
print("#-- Sample 5-1 --#")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Cannot find the file
#df = pd.read_csv('https://bit.ly/3go0Ant', delimiter=",") # Not found
#df = pd.read_csv('Https://bit.ly/3cIH97A', delimiter=",") # OK
df = pd.read_csv('sample5-2_data.csv', delimiter=",") # 課本範例資料應該是這個才對

X = df.values[:, :-1]
Y = df.values[:, -1]
#print("X= ", X)
#print("Y= ", Y)

fit = LinearRegression().fit(X, Y)

m = fit.coef_.flatten()
b = fit.intercept_.flatten()
print("m = {0}".format(m))
print("b = {0}".format(b))

plt.plot(X, Y, 'o')
plt.plot(X, m*X+b)
plt.show()



#-- Sample 5-2 --#
print("#-- Sample 5-2 --#")
import pandas as pd

# Cannot find the file
#df = pd.read_csv('https://bit.ly/3go0Ant', delimiter=",") # Not found
#df = pd.read_csv('Https://bit.ly/3cIH97A', delimiter=",") # OK
points = pd.read_csv('sample5-2_data.csv', delimiter=",").itertuples()

m = 1.93939394
b = 4.73333333

# 計算殘差
for p in points :
    y_actual = p.y
    y_predict = m * p.x + b
    residual = y_actual - y_predict
    print(residual)


#-- Sample5-4 --#
print("#-- Sample5-4 --#")
import pandas as pd

#df = pd.read_csv('https://bit.ly/2KF29Bd', delimiter=",") # OK
points = pd.read_csv('sample5-2_data.csv', delimiter=",").itertuples()

m = 1.93939
b = 4.73333

sum_of_squares = 0.0

#計算平方和
for p in points :
    y_actual = p.y
    y_predict = m * p.x + b
    residual_squared = (y_predict - y_actual)**2
    sum_of_squares += residual_squared
    
print("sum of squares = {}".format(sum_of_squares)) # 28.096969704500005


#-- Sample 5-5 --#
print("#-- Sample 5-5 --#")
import pandas as pd

points = list(pd.read_csv('https://bit.ly/2KF29Bd', delimiter=",").itertuples())
#points = pd.read_csv('sample5-2_data.csv', delimiter=",").itertuples()

n = len(points)

m = (n*sum(p.x*p.y for p in points) - sum(p.x for p in points) *
    sum(p.y for p in points)) / (n*sum(p.x**2 for p in points) -
    sum(p.x for p in points)**2)
b = (sum(p.y for p in points) / n) - m * sum(p.x for p in points) / n

print(m, b) # 1.9393939393939394 4.7333333333333325


#-- Sample 5-6 --#
print("#-- Sample 5-6 --#")
import pandas as pd
from numpy.linalg import inv
import numpy as np

# 匯入資料點
df = pd.read_csv('https://bit.ly/2KF29Bd', delimiter=",")

# 提取輸入變數(所有列、除了最後一行之外的所有行)
X = df.values[:, :-1].flatten()
# 添加占位符"1"行來產生截距
X_1 = np.vstack([X, np.ones(len(X))]).T
# 提取輸出行(所有列、最後一行)
Y = df.values[:, -1]

# 計算斜率和截距係數
b = inv(X_1.transpose() @ X_1) @ (X_1.transpose() @ Y)
print(b)

#預測y值
y_predict = X_1.dot(b)


#-- Sample5-7 --#
print("#-- Sample5-7 --#")
import pandas as pd
from numpy.linalg import qr, inv
import numpy as np

df = pd.read_csv('https://bit.ly/2KF29Bd', delimiter=",")

X = df.values[:, :-1].flatten()
X_1 = np.vstack([X, np.ones(len(X))]).transpose()
Y= df.values[:, -1]

Q, R = qr(X_1)
b = inv(R).dot(Q.transpose()).dot(Y)

print(b) # [1.93939394 4.73333333]







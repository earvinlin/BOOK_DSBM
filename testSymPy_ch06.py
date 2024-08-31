import math

#-- Sample 6-1 --#
print("#-- Sample 6-1 --#")
def predict_probability(x, b0, b1) :
    p = 1.0 / (1.0 + math.exp(-(b0 + b1 * x)))
    return p


#-- Sample 6-2 --#
print("\n#-- Sample 6-2 --#")
from sympy import *

b0, b1, x = symbols('b0 b1 x')
p = 1.0 / (1.0 + exp(-(b0 + b1 * x)))

p = p.subs(b0, -2.823)
p = p.subs(b1, 0.62)

print(p)
plot(p)


#-- Sample 6-3 --#
print("\n#-- Sample 6-3 --#")
# 在Scikit-learn中使用簡單的邏輯迴歸
import pandas as pd
from sklearn.linear_model import LogisticRegression

#df = pd.read_csv('https://bit.ly/33ebs2R', delimiter=",")
df = pd.read_csv('sample6-3_data.csv', delimiter=",")

# 提取輸入變數(所有列，除了最後一行的所有行)
X = df.values[:, :-1]
# 提取輸出變數(所有列、最後一行)
Y = df.values[:, -1]

# 執行邏輯迴歸
# 關閉懲罰
model = LogisticRegression(penalty='none')
model.fit(X, Y)

# 印出beta1
print(model.coef_.flatten())      # 0.69267212
#印出beta0
print(model.intercept_.flatten()) # -3.17576395












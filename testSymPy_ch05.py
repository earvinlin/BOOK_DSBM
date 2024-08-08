#-- Sample 5-1 --#
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Cannot find the file
#df = pd.read_csv('https://bit.ly/3go0Ant', delimiter=",")

#df = pd.read_csv('Https://bit.ly/3cIH97A', delimiter=",")
df = pd.read_csv('sample5-1_data.csv', delimiter=",")

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



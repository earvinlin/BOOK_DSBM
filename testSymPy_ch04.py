from sympy import *
from sympy.plotting import plot3d
from scipy.stats import binom
from scipy.stats import beta
import numpy as np

#-- Sample 4-8 --#
from numpy import array

v = array([1, 1])
i_hat = array([2, 1])
j_hat = array([0, 3])

print("=== case 1 ===")
basis = array([i_hat, j_hat])
print(basis)
new_v0 = basis.dot(v)
print(new_v0)


"""
numpy.transpose(a, axes=None)
a : input array, 輸入數組
axes : 可選，整型list。默認情況下，反轉維度，
       否則根據給定的值對軸進行排列。
"""
a1 = array([[1,2,3], [4,5,6]])
print("Before transpose: ", a1)
a2 = a1.transpose()
print("After transpose: ", a2)


print("=== case 2 ===")
basis = array([i_hat, j_hat]).transpose()
print(basis)
basis.dot(v)
new_v1 = basis.dot(v)
print(new_v1)

#-- Sample 4-5 --#
v = array([3,2])
w = array([[2],[-1]])
print("v data: ", v, ", v shap: ", v.shape)
print("w data: ", w, ", w shap: ", w.shape)

print("------------------")

v1 = np.transpose(v)
print("v1 transpose: ", v1)
print("v1 transpose shape: ", v1.shape)

#v_plus_w = v + w
#print("v+w= ", v_plus_w)

#-- Sample 4-11 --#
print("\n#-- Sample 4-11 --#")
from numpy import array
i_hat1 = array([0, 1])
j_hat1 = array([-1, 0])
transform1 = array([i_hat1, j_hat1]).transpose()
print("transform1= ", transform1)

i_hat2 = array([1, 0])
j_hat2 = array([1, 1])
transform2 = array([i_hat2, j_hat2]).transpose()
print("transform2= ", transform2)

combined = transform2 @ transform1

print("COMBINED MATRIX:\n {}".format(combined))

v = array([1, 2])
print("combined.dot(v) : ",combined.dot(v)) # [-1, 1]



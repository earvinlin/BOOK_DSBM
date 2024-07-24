from sympy import *
from sympy.plotting import plot3d
from scipy.stats import binom
from scipy.stats import beta

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



print("=== case 2 ===")
basis = array([i_hat, j_hat]).transpose()
print(basis)
basis.dot(v)
new_v1 = basis.dot(v)
print(new_v1)



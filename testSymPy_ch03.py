from sympy import *
from sympy.plotting import plot3d
from scipy.stats import binom
from scipy.stats import beta

#-- Sample 3-9 機率密度函數(PDF) --#
def normal_pdf(x: float, mean: float, std_dev: float) -> float :
    return (1.0 / (2.0 * math.pi * std_dev ** 2) ** 0.5) * \
        math.exp(-1.0 *((x - mean) ** 2 / (2.0 * std_dev ** 2)))



"""
#-- Sample 3-15 --#
import random
import plotly.express as px

sample_size = 31
sample_count = 1000

# 中央極限定理，1000組樣本，每組包含31個
#介於 0.0 ~ 1.0之間的亂數
# i start with 0
x_values = [(sum([random.uniform(0.0, 1.0) for i in range(sample_size)]) / \
             sample_size) for _ in range(sample_count)]

y_values = [1 for _ in range(sample_count)]

px.histogram(x=x_values, y=y_values, nbins=20).show()
"""

#-- Sample 3-16 --#
print("#-- Sample 3-16 --#")

from scipy.stats import norm

def critical_z_value(p) :
    norm_dist = norm(loc = 0.0, scale = 1.0)
    left_tail_area = (1.0 - p) / 2.0
    upper_area = 1.0 - ((1.0 - p) / 2.0)
    return norm_dist.ppf(left_tail_area), norm_dist.ppf(upper_area)

print(critical_z_value(p = .95))
# (-1.959963984540054, 1.959963984540054)


print("\n\n\n")
print("==============")
print("\n")
#-- Testing Area --#
import random

x_values = [(sum([random.uniform(0.0, 1.0) for i in range(31)]) / 31) for _ in range(10)]
#           ---------------------------------------------------------
print(x_values)
for i in range(11) :
    print(i)

y_values = [1 for _ in range(30)]
print(y_values)
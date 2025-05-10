#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

# your code here
x = np.arange(0, 11)
plt.plot(x, y, 'r-')  # 'r-' means red solid line
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, 10)
plt.grid(True)
plt.show()


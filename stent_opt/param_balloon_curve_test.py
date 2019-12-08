import matplotlib.pyplot as plt
import numpy

import math

def f(a):
    x = [0.0, 1.0, 2.0 + a, 3.0-a, 4.0, 5.0]
    y = [0.0, 0.0, 0.1*a, -0.1*a, 0.0, 0.0]

    return x, y


if __name__ == "__main__":
    x, y = f(2.0)
    plt.plot(x, y, '-o')
    plt.show()
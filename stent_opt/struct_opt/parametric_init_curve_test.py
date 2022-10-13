import typing
import math

import numpy

import matplotlib.pyplot as plt


class P(typing.NamedTuple):
    x: float
    y: float


def func_eg(t):
    return P(t, t**2)


def func_a(t):
    t1 = math.pi * 2 * t
    return P(t + math.sin(2 * t1) / math.pi / 2, 0.5 * (1 - math.cos(t1)))


def plot_curve(f):
    t_range = numpy.linspace(0, 1, 200)
    ps = [f(t) for t in t_range]
    x = [p.x for p in ps]
    y = [p.y for p in ps]

    plt.plot(x, y)
    plt.title(f.__name__)
    plt.show()


if __name__ == "__main__":
    plot_curve(func_a)

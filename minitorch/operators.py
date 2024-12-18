"""Collection of the core mathematical operators used throughout the code base."""
import math
from typing import Callable, Iterable


def mul(x, y):
    """ Multiplies two numbers

    参数:
        x: 输入数值
        y: 输入数值

    返回: x * y
    """
    return x * y


def id(x):
    """ Returns the input unchanged

    参数:
        x: 输入数值

    返回: x
    """
    return x


def add(x, y):
    """ Adds two numbers

    参数:
        x: 输入数值
        y: 输入数值

    返回: x + y
    """
    return x + y


def neg(x):
    """ Negates a number

    参数:
        x: 输入数值

    返回: -x
    """
    return -x


def lt(x, y):
    """ Check if one number is less than another

    参数:
        x: 输入数值
        y: 输入数值

    返回: True if x < y else False
    """
    return x < y


def eq(x, y):
    """ Check if two number are equal

    参数:
        x: 输入数值
        y: 输入数值

    返回: True if x == y else False
    """
    return x == y


def max(x, y):
    """ Return the larger of two numbers

    参数:
        x: 输入数值
        y: 输入数值

    返回: x if x >= y else y
    """
    if (x >= y):
        return x
    else:
        return y


def is_close(x, y):
    """ Check if two numbers are close in value

    参数:
        x: 输入数值
        y: 输入数值

    返回: True if |x - y| < 1e-5 else False
    """
    return abs(x - y) < 1e-5


def sigmoid(x):
    """ Calvulate the sigmoid function

    参数:
        x: 输入数值

    返回: sigmoid(x)
    """
    return 1.0 / (1.0 + math.exp(-x))


def relu(x):
    """ Applies the ReLU activation function

    参数:
        x: 输入数值

    返回: ReLU(x)
    """
    return max(0, x)


def log(x):
    """ Calvulate the nature logarithm function

    参数:
        x: 输入数值

    返回: log(x)
    """
    return math.log(x)


def exp(x):
    """ Calvulate the xponential f function

    参数:
        x: 输入数值

    返回: exp(x)
    """
    return math.exp(x)


def inv(x):
    """ Calculates the reciprocal function

    参数:
        x: 输入数值

    返回: inv(x)
    """
    if x == 0:
        raise ValueError("Logarithm input must be not equal 0.")
    return 1 / x


def log_back(x, upstream_grad):
    """ Computes the derivative of log times a second arg

    参数:
        x: 输入数值

    返回: upstream_grad * (1 / x)
    """
    if x <= 0:
        raise ValueError("Logarithm input must be greater than 0.")
    return upstream_grad * (1 / x)


def inv_back(x, upstream_grad):
    """ Computes the derivative of reciprocal times a second arg

    参数:
        x: 输入数值

    返回: upstream_grad * (-1 / (x ** 2))
    """
    if x == 0:
        raise ValueError("Input x must not be zero.")
    return upstream_grad * (-1 / (x ** 2))


def relu_back(x, upstream_grad):
    """ Computes the derivative of relu times a second arg

    参数:
        x: 输入数值

    返回: upstream_grad if x > 0 else 0
    """
    if x > 0:
        return upstream_grad
    else:
        return 0


def map(func: Callable) -> Callable:
    """ applies a given function to each element of an iterable """
    def func_builtin(list: Iterable) -> Iterable:
        return [func(x) for x in list]
    return func_builtin


def zipWith(func: Callable) -> Callable:
    """ combines elements from two iterables using a given function """
    def func_builtin(list1: Iterable, list2: Iterable) -> Iterable:
        return [func(x, y) for x, y in zip(list1, list2)]
    return func_builtin


def reduce(func: Callable, start) -> Callable:
    """ reduces an iterable to a single value using a given function """
    def process(ls):
        ans = start
        for item in ls:
            ans = func(ans, item)
        return ans
    return process


def negList(l: Iterable) -> Iterable:
    """ negate a list """
    func = map(neg)
    return func(l)


def addLists(l1: Iterable, l2: Iterable) -> Iterable:
    """ add twp Lists together"""
    func = zipWith(add)
    return func(l1, l2)


def sum(l: Iterable) -> Iterable:
    """ sum list """
    func = reduce(add, 0)
    return func(l)


def prod(l: Iterable) -> Iterable:
    """ take the product of lists """
    func = reduce(mul, 1)
    return func(l)

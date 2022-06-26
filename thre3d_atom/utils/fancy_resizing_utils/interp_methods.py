"""
---------------------------------------------------------------------------------------
|                                !!! ORIGINAL LICENSE !!!                             |
---------------------------------------------------------------------------------------
|    MIT License                                                                      |
|                                                                                     |
|    Copyright (c) 2020 Assaf Shocher                                                 |
|                                                                                     |
|    Permission is hereby granted, free of charge, to any person obtaining a copy     |
|    of this software and associated documentation files (the "Software"), to deal    |
|    in the Software without restriction, including without limitation the rights     |
|    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell        |
|    copies of the Software, and to permit persons to whom the Software is            |
|    furnished to do so, subject to the following conditions:                         |
|                                                                                     |
|    The above copyright notice and this permission notice shall be included in all   |
|    copies or substantial portions of the Software.                                  |
|                                                                                     |
|    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR       |
|    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,         |
|    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE      |
|    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER           |
|    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,    |
|    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE    |
|    SOFTWARE.                                                                        |
---------------------------------------------------------------------------------------

This code has been adapted from: https://github.com/assafshocher/ResizeRight
"""
# TODO: The code needs understanding and cleaning!
#  Time to take out the signal-processing gloves :smile:
from math import pi

try:
    import torch
except ImportError:
    torch = None

try:
    import numpy
except ImportError:
    numpy = None

if numpy is None and torch is None:
    raise ImportError("Must have either Numpy or PyTorch but both not found")


def set_framework_dependencies(x):
    if type(x) is numpy.ndarray:
        to_dtype = lambda a: a
        fw = numpy
    else:
        to_dtype = lambda a: a.to(x.dtype)
        fw = torch
    eps = fw.finfo(fw.float32).eps
    return fw, to_dtype, eps


def support_sz(sz):
    def wrapper(f):
        f.support_sz = sz
        return f

    return wrapper


@support_sz(4)
def cubic(x):
    fw, to_dtype, eps = set_framework_dependencies(x)
    absx = fw.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5 * absx3 - 2.5 * absx2 + 1.0) * to_dtype(absx <= 1.0) + (
        -0.5 * absx3 + 2.5 * absx2 - 4.0 * absx + 2.0
    ) * to_dtype((1.0 < absx) & (absx <= 2.0))


@support_sz(4)
def lanczos2(x):
    fw, to_dtype, eps = set_framework_dependencies(x)
    return (
        (fw.sin(pi * x) * fw.sin(pi * x / 2) + eps) / ((pi**2 * x**2 / 2) + eps)
    ) * to_dtype(abs(x) < 2)


@support_sz(6)
def lanczos3(x):
    fw, to_dtype, eps = set_framework_dependencies(x)
    return (
        (fw.sin(pi * x) * fw.sin(pi * x / 3) + eps) / ((pi**2 * x**2 / 3) + eps)
    ) * to_dtype(abs(x) < 3)


@support_sz(2)
def linear(x):
    fw, to_dtype, eps = set_framework_dependencies(x)
    return (x + 1) * to_dtype((-1 <= x) & (x < 0)) + (1 - x) * to_dtype(
        (0 <= x) & (x <= 1)
    )


@support_sz(1)
def box(x):
    fw, to_dtype, eps = set_framework_dependencies(x)
    return to_dtype((-1 <= x) & (x < 0)) + to_dtype((0 <= x) & (x <= 1))

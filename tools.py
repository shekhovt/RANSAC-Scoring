from typing import Tuple, Callable, Any
import os
import sys
import math
import pickle


def mkdir_recursive(path):
    sub_path = os.path.dirname(path)
    if len(sub_path) > 0 and not os.path.exists(sub_path):
        mkdir_recursive(sub_path)
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except FileExistsError:  # it could have been created in the meantime by a parallel process
            pass


def force_path(file_name):
    mkdir_recursive(os.path.dirname(file_name))


def save_object(filename, obj):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as inp:
        try:
            obj = pickle.load(inp)
        except:
            # import pickle5 as pickle
            obj = pickle.load(inp)
    return obj


class dotdict(dict):
    __getattr__ = dict.get
    __getitem__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)


def format_digits(x, sig_digits):
    # digits = -int(math.floor(math.log10(abs(x)))) + sig_digits - 1
    # return '{num:.{digits}f}'.format(num=x, digits=digits)
    return '{num:.{digits}g}'.format(num=x, digits=sig_digits)


def format_std(x, std=0.0, std1=0.0, units=''):
    """ given a value and its std generate string like 0.23 ± 0.01 with significant digits determined by std"""
    ci = math.sqrt(std ** 2 + std1 ** 2)
    stddigits = 1
    if abs(x) == 0.0 or math.isnan(x) or math.isinf(x):
        digit0 = 0
    else:
        digit0 = int(math.floor(math.log10(abs(x))))  # decimal digit of x
    if ci > 0:
        digit = int(math.floor(math.log10(ci))) - 1
        if std >= 10:
            digit -= 1
            stddigits += 1
        sdigits = digit0 - digit  # take digits to precision of ci
    else:
        sdigits = 3
        # take 3 significant digits of x
    sdigits = max(sdigits, 0)  # otherwise format does not work
    sdigits += 1
    stddigits += 1
    # sdigits = max(sdigits, digit0)  # otherwise format does not work
    s = '{num:.{digits}g}'.format(num=x, digits=sdigits)
    l = 8
    if std > 0:
        s += ' ±{num:.{digits}g}'.format(num=std, digits=stddigits)
        l += 6
    if std1 > 0:
        s += ' ±{num:.{digits}g}'.format(num=std1, digits=stddigits)
        l += 6
    s = s + units
    s = s.ljust(l)
    return s

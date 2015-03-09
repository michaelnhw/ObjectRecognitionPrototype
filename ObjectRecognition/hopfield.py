from numpy import *
from pylab import imshow, cm, show
import scipy.io

def train(pattern):
    pattern_length = shape(pattern)[1]
    data = pattern-0.5
    weights = dot(data, transpose(data))
    weights = weights * 4
    weights[diag_indices(pattern_length)] = 0
    return weights

def retrive(weights, pattern, steps):
    data = pattern
    for i in range(steps):
        temp = dot(weights,data)
        temp = temp > 0
        data = 1 * temp
    return data

def to_pattern(letter):
    return array([+1 if c=='X' else 0 for c in letter.replace('\n','')])

def display(pattern):
    imshow(pattern.reshape((5,5)),cmap=cm.binary, interpolation='nearest')
    show()

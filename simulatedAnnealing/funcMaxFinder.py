import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

MAX_TIME = 500
NR_ITERATION = 300
CONST_INIT_TEMP = 1000
CONST_ALPHA = 0.95

# def func(x1, x2, x3, x4, x5, x6):
#     return 21.5+x1*np.sin(4*np.pi*x1)+x2*np.sin(20*np.pi*x2)+x3*np.sin(7*np.pi*x3)+x4*np.sin(11*np.pi*x4)+x5*np.sin(8*np.pi*x5)+x6*np.sin(6*np.pi*x6)

def func(x1, x2):
    return 21.5+x1*np.sin(4*np.pi*x1)+x2*np.sin(20*np.pi*x2)

def startval(intv):
    if intv.left is -np.inf:
        return intv.right - 1
    elif intv.right is np.inf:
        return intv.left + 1
    else:
        r = np.random.random()
        return np.dot([r,1-r], [intv.left,intv.right])

def fixval(col):
    old, new, intv = col
    if new in intv:
        return new
    elif new <= intv.left:
        r = np.random.random()
        return np.dot([r,1-r], [intv.left,old])
    else:
        r = np.random.random()
        return np.dot([r,1-r], [intv.right,old])

def funcMax(func, validIntv):
    argcount = func.__code__.co_argcount
    # valid interval preprocessing
    validIntv = validIntv[0:argcount]
    fill = [pd.Interval(-np.inf,np.inf)] * (argcount-len(validIntv))
    validIntv.extend(fill)
    validIntv = np.array(validIntv)
    # generate initial value
    solution = np.vectorize(startval)(validIntv)
    y = func(*solution)
    # for t in tqdm(range(MAX_TIME)):
    Y = []
    exactSearch = 0.01
    for t in range(MAX_TIME):
        for i in range(NR_ITERATION):
            r = np.random.normal(size=argcount)
            b = r / np.sqrt(r**2)
            T = CONST_INIT_TEMP*CONST_ALPHA**t
            # exactSearch = np.sqrt(T)
            exactSearch = t
            newS = r + exactSearch * b
            newS = np.apply_along_axis(fixval, axis=0, arr=np.vstack((solution, newS, validIntv)))
            newy = func(*newS)
            p = np.exp((newy-y)/(CONST_INIT_TEMP*CONST_ALPHA**t)) # if newy > y then p > 1
            if np.random.random() < p:
                solution = newS
                y = newy
                # exactSearch = 0.01
            else:
                # exactSearch += 0.01 / exactSearch
                pass
        print(f'{t}/{MAX_TIME}, y={y}, eS={exactSearch}', end='\r')
        Y.append(y)
    print()
    plt.plot(np.arange(MAX_TIME), Y)
    for i in range(argcount):
        print(f'{func.__code__.co_varnames[i]} = {solution[i]}')
    print(y)
    plt.show()
    return y

if __name__ == "__main__":
    x1Intv = pd.Interval(-3,12.1,closed='both')
    x2Intv = pd.Interval(4.1,5.8,closed='both')
    # x3Intv = pd.Interval(1.2,4.9,closed='both')
    # x4Intv = pd.Interval(-1,4,closed='both')
    # x5Intv = pd.Interval(0,6.6,closed='both')
    # x6Intv = pd.Interval(3.6,10,closed='both')
    # print(funcMax(func, [x1Intv, x2Intv, x3Intv, x4Intv, x5Intv, x6Intv]))
    funcMax(func, [x1Intv, x2Intv])
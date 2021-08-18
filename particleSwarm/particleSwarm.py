import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def func(x1, x2):
    return 21.5+x1*np.sin(4*np.pi*x1)+x2*np.sin(20*np.pi*x2)
    # return -(x1**2 + x2**2 + 3*x1)

omgea = 0.5
phi_p = 0.1
phi_g = 0.4

NR_PARTICLES = 1000
NR_DIMENSION = None
NR_ITERATION = 25
CONST_LEARNING_RATE = 1

v = None
x = None
p = None
g = None

def initParticles(func, validIntv):
    global v, x, p, g, NR_DIMENSION
    NR_DIMENSION = func.__code__.co_argcount
    low = np.array([intv.left for intv in validIntv])
    high = np.array([intv.right for intv in validIntv])
    x = np.random.uniform(low, high, size=(NR_PARTICLES, NR_DIMENSION))
    p = x.copy()
    v_range = np.abs(low - high)
    v = np.random.uniform(-v_range, v_range, size=(NR_PARTICLES, NR_DIMENSION))
    f = np.array([func(*l) for l in p])
    g = p[np.argmax(f)]

def particleSwarm(func, validIntv):
    initParticles(func, validIntv)
    global v, x, p, g, NR_DIMENSION
    Y = []
    low = np.array([intv.left for intv in validIntv])
    high = np.array([intv.right for intv in validIntv])
    for _ in tqdm(range(NR_ITERATION)):
        for i in range(NR_PARTICLES):
            r_p = np.random.random(NR_DIMENSION)
            r_g = np.random.random(NR_DIMENSION)
            # import pdb;pdb.set_trace()
            v[i] = omgea*v[i] + phi_p*r_p*(p[i]-x[i]) + phi_g*r_g*(g-x[i])
            x[i] = x[i] + CONST_LEARNING_RATE*v[i]
            x[i] = np.where(x[i]<low, low, x[i])
            x[i] = np.where(x[i]>high, high, x[i])
            if func(*x[i]) > func(*p[i]):
                p[i] = x[i]
                if func(*p[i]) > func(*g):
                    g = p[i]
        Y.append(func(*g))
    print(g)
    print(func(*g))
    plt.plot(np.arange(0,NR_ITERATION), Y)
    plt.show()
    plt.scatter(x.T[0], x.T[1], s=1, color='b')
    plt.show()

if __name__ == "__main__":
    x1Intv = pd.Interval(-3,12.1,closed='both')
    x2Intv = pd.Interval(4.1,5.8,closed='both')
    # x1Intv = pd.Interval(-10,10,closed='both')
    # x2Intv = pd.Interval(-10,10,closed='both')
    particleSwarm(func, [x1Intv, x2Intv])
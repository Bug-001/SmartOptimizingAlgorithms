# 任务分配均衡
# M个服务节点、N个任务，问如何分配可使总时长最小

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

servers = None
tasks = None
m = None
n = None

NR_CHROMOSOME = 40
NR_ITERATION = 1000
RATE_EXCELLENT = 0.30
RATE_MUTATE = 0.05

def initChromosome():
    return np.random.randint(m, size=(NR_CHROMOSOME, n))

def calAnswer(ch):
    lapse = np.zeros(m)
    for i in range(n):
        lapse[ch[i]] += tasks[i] / servers[ch[i]]
    return np.max(lapse)

def calAdaptation(ans):
    u, l = np.max(ans), np.min(ans)
    return (u-l)*(u-ans)

def cross(ch1, ch2):
    return np.where(np.random.rand(n) < 0.5, ch1, ch2)

def mutate(ch):
    return np.where(np.random.rand(n) < RATE_MUTATE, np.random.randint(m), ch)

def geneticAlg():
    # init chromosome
    chromosome = initChromosome()
    for i in tqdm(range(NR_ITERATION)):
        # calculate the elapsed time corresponding to the certain chromosome
        ans = np.apply_along_axis(calAnswer, axis=1, arr=chromosome)
        # drawing
        plt.scatter(i*np.ones(NR_CHROMOSOME), ans, s=1, color='b')
        # calculate adaptation
        adaption = calAdaptation(ans)
        # save excellent chromosome
        exRank = int(NR_CHROMOSOME * RATE_EXCELLENT)
        nextGeneration = chromosome[np.argsort(-adaption)[:exRank]]
        selectRate = adaption / np.sum(adaption)
        for i in range(NR_CHROMOSOME - exRank):
            # choose, cross, and mutate
            ch1, ch2 = chromosome[np.random.choice(np.arange(0, NR_CHROMOSOME), size=2, replace=False, p=selectRate)]
            newchrome = mutate(cross(ch1, ch2))
            nextGeneration = np.r_[nextGeneration, [newchrome]]
        # update chromosome of new generation
        chromosome = nextGeneration
    # draw converging scatter plot and return optimized slolution
    ret = chromosome[np.argmin(np.apply_along_axis(calAnswer, axis=1, arr=chromosome))]
    print(calAnswer(ret))
    plt.show()
    return ret
    

if __name__ == "__main__":
    # performance data of m servers
    servers = np.load('mDat.npy')
    # time of n tasks to be done
    tasks = np.load('nDat.npy')
    m = servers.shape[0]
    n = tasks.shape[0]
    # return an optimized solution
    ans = geneticAlg()
    np.save('ans.npy', ans)

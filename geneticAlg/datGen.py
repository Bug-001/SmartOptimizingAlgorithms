import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data generator for load balance problem")
    parser.add_argument('-M', metavar='server count', type=int)
    parser.add_argument('-N', metavar='task count', type=int)
    args = parser.parse_args()
    M = args.M
    N = args.N
    mDat = np.random.randint(100, size=M)
    nDat = np.random.rand(N) * 2000
    np.save('mDat.npy', mDat)
    np.save('nDat.npy', nDat)

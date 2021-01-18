import multiprocessing as mp
import time
import sys
import argparse
import os
import numpy as np


def Loop(map, rev_map, seed_set, n):
    cnt, total = 0, 0
        # print(x)
    res = LT(map, rev_map, seed_set, n)
    total += res
    return total


def LT(map, rev_map, seed, n):
    thresh = np.random.rand(n+1)
    RR = seed.copy()
    active = seed.copy()
    while len(active) != 0:
        newAck = []
        for i in active:
            for pair in map[i]:
                neighbor = pair[0]
                if neighbor not in RR and getTotalWeight(rev_map, neighbor, RR) >= thresh[neighbor]:
                    RR.append(neighbor)
                    newAck.append(neighbor)
        active = newAck
    return len(RR)

def getTotalWeight(map, node, RR):
    thresh = 0
    for pair in map[node]:
        if pair[0] in RR:
            thresh += pair[1]
    return thresh
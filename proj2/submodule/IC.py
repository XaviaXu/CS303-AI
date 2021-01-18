import multiprocessing as mp
import time
import sys
import argparse
import os
import numpy as np


def Loop(map, rev_map, seed_set, n):
    cnt, total = 0, 0


    res = IC(map, rev_map, seed_set, n)
    total += res

        #print(res)

    return total


def IC(map, rev_map, seed, n):
    RR = seed.copy()
    active = seed.copy()
    while len(active) != 0:
        newAck = []
        for seed in active:
            for pair in map[seed]:
                neighbor, weight = pair
                if neighbor not in RR:
                    if np.random.random() <= weight:
                        RR.append(neighbor)
                        newAck.append(neighbor)
        active = newAck
    return len(RR)

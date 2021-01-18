import multiprocessing as mp
import time
import sys
import argparse
import os
import numpy as np


def Loop(map, rev_map, seed_set, n, lasting_time):
    cnt, total = 0, 0
    sta = time.time()
    end = sta
    while end - sta < lasting_time:
        total += IC(map, rev_map, seed_set, n)
        cnt += 1
        end = time.time()

    return total / cnt


def IC(map, rev_map, seed_set, n):
    Active = seed_set.copy()

    cnt = len(Active)
    isVisited = []
    for i in Active:
        isVisited.append(i)

    while len(Active) != 0:
        newAct = []
        for seed in Active:
            for neighbor in map[seed]:
                if neighbor not in isVisited:
                    if np.random.random() <= 1 /len(rev_map[neighbor]):
                        isVisited.append(neighbor)
                        newAct.append(neighbor)
        cnt += len(newAct)
        Active = newAct.copy()
    return cnt

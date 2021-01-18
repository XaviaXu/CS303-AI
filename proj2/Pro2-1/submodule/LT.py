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
        # print(x)
        total += LT(map, rev_map, seed_set, n)
        cnt += 1
        end = time.time()

    return total / cnt


def LT(map, rev_map, seed_set, n):
    Active = seed_set.copy()
    thresh = np.random.rand(n+1)

    # thresholds?
    cnt = len(Active)
    isVisited = []
    for i in Active:
        isVisited.append(i)
    while len(Active) != 0:
        newAct = []
        for seed in Active:
            for neighbor in map[seed]:
                if neighbor not in isVisited:
                    if isAffected(rev_map, isVisited, neighbor, thresh):
                        isVisited.append(neighbor)
                        newAct.append(neighbor)
        cnt += len(newAct)
        Active = newAct.copy()
    return cnt


def isAffected(rev_map, isVisited, neighbor, threshold):
    inEdge = rev_map[neighbor]
    cnt = 0
    for i in inEdge:
        if i in isVisited:
            cnt += 1
    if cnt / len(inEdge) >= threshold[int(neighbor)]:
        return True
    return False

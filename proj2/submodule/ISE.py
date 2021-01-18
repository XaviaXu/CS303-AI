import multiprocessing as mp
import time
import sys
import argparse
import os
import numpy as np
from submodule import LT, IC

core = 7

map = {}
rev_map = {}
seed = []


def readNet(path):
    global n, v
    f = open(path,'r',encoding='utf-8')
    temp = f.readlines()
    for i in temp:
        edge = i.strip('\n').split(' ')
        if len(edge) == 2:
            n, v = eval(edge[0]), eval(edge[1])
            initial(n)
        else:
            sta, end = edge[0], edge[1]
            map[sta].append(end)
            rev_map[end].append(sta)
    return n, v


def readSeed(path):
    f = open(path,'r',encoding='utf-8')
    temp = f.read()
    li = temp.split("\n")
    return li

def initial(n):
    for i in range(1,n+1):
        map[str(i)] = []
        rev_map[str(i)] = []




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default='netHEPT.txt')
    parser.add_argument('-s', '--seed', type=str, default='network_seeds.txt')
    parser.add_argument('-m', '--model', type=str, default='IC')
    parser.add_argument('-t', '--time_limit', type=int, default=60)
    sta = time.time()
    args = parser.parse_args()
    file_name = os.path.abspath(args.file_name)
    seed_path = os.path.abspath(args.seed)
    model = args.model
    time_limit = args.time_limit

    read = time.time()
    n, v = readNet(file_name)
    seeds = readSeed(seed_path)
    proc = time.time()
    #print("Processing data:"+str(proc-read))


def run(map,rev_map,seeds,n,model):

    pool = mp.Pool(core)
    result = []

    if model == 'LT':
        for i in range(core):
            result.append(pool.apply_async(LT.Loop, args=(map, rev_map, seeds, n)))
    else:
        for i in range(core):
            result.append(pool.apply_async(IC.Loop, args=(map, rev_map, seeds, n)))


    pool.close()
    pool.join()

    #print(end - sta)

    total = 0
    for i in result:
        total += i.get()
    print(total/core)



    '''
    程序结束后强制退出，跳过垃圾回收时间, 如果没有这个操作会额外需要几秒程序才能完全退出
    '''
    sys.stdout.flush()

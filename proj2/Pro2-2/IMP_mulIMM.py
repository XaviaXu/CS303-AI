import multiprocessing as mp
import time
import sys
import argparse
import os
import numpy as np
import math

core = 7
Workers = []
weight_min = 0.1
init = 0




class MyProcess(mp.Process):
    def __init__(self):
        super(MyProcess, self).__init__(target=self.start)
        self.inQueue = mp.Queue()
        self.outQueue = mp.Queue()
        self.func = func
        self.n = n
        self.map = map
        self.rev_map = rev_map

    def run(self):
        while True:
            loop = self.inQueue.get()
            R = []
            for i in range(loop):
                seed = np.random.RandomState().randint(1, self.n+1)
                #print(seed)
                rr = self.func(self.map, self.rev_map, seed)
                R.append(rr)
            self.outQueue.put(R)


def createProcess():
    for i in range(core):
        worker = MyProcess()
        worker.start()
        Workers.append(worker)


def close():
    for worker in Workers:
        worker.terminate()
        worker.join()


def initial(dic):
    for i in range(init, n + 1):
        dic[i] = []


def readNet(path):
    global n, v
    f = open(path, 'r', encoding='utf-8')
    temp = f.readlines()
    for i in temp:
        edge = i.strip('\n').split(' ')
        if len(edge) == 2:
            n, v = eval(edge[0]), eval(edge[1])
            initial(map)
            initial(rev_map)
        else:
            sta, end, weight = int(edge[0]), int(edge[1]), eval(edge[2])
            if weight >= weight_min:
                map[sta].append((end, weight))
                rev_map[end].append((sta, weight))
    return n, v


def nodeSelection(R, k):
    sk = []
    cover = np.zeros(n + 1, int)
    coveredRR = {}
    coveredRange = 0
    initial(coveredRR)
    # 初始化 找出覆盖范围
    for i in range(len(R)):
        RR_set = R[i]
        for j in RR_set:
            cover[j] += 1
            coveredRR[j].append(i)

    for i in range(k):
        max_v = np.where(cover == np.max(cover))[0][0]
        coveredRange += max(cover)
        sk.append(max_v)

        while len(coveredRR[max_v]) != 0:
            j = coveredRR[max_v][0]
            for node in R[j]:
                cover[node] -= 1
                coveredRR[node].remove(j)
    return sk, coveredRange / len(R)


def Sampling(k, e, l):
    R = []
    LB = 1
    e_p = e * math.sqrt(2)
    lambda_p = ((2 + 2 * e_p / 3) * n * (log_nk(n, k) + l * math.log(n)
                                         + math.log(math.log2(n)))) / pow(e_p, 2)
    for i in range(1, int(math.log2(n - 1) + 1)):
        x = n / math.pow(2, i)
        theta_i = lambda_p / x
        # while len(R) <= theta_i:
        #     # generate RR
        #     R.append(generateRR())
        diff = math.ceil(theta_i - len(R))
        delta = int((theta_i - len(R)) / core)

        for p in range(core - 1):
            Workers[p].inQueue.put(delta)
        Workers[core - 1].inQueue.put(diff - delta * (core - 1))
        for p in Workers:
            res = p.outQueue.get()
            R += res

        si, Fr = nodeSelection(R, k)
        if n * Fr >= (1 + e_p) * x:
            LB = n * Fr / (1 + e_p)*0.8
            break
    alpha = (l * math.log(n) + math.log(2)) ** 0.5
    beta = ((1 - 1 / math.e) * (log_nk(n, k) + l * math.log(n) + math.log(2))) ** 0.5
    lambda_sta = 2 * n * math.pow(((1 - 1 / math.e) * alpha + beta), 2) * math.pow(e, -2)
    theta = lambda_sta / LB

    # while len(R) <= theta:
    #     # generate RR
    #     R.append(generateRR())
    delta = math.ceil((theta - len(R)) / core)
    for p in Workers:
        p.inQueue.put(delta)
    for p in Workers:
        res = p.outQueue.get()
        R += res


    return R


def log_nk(n, k):
    result = 0
    for i in range(1, n + 1):
        result += math.log(i)
    for i in range(1, k + 1):
        result -= math.log(i)
    for i in range(1, n - k + 1):
        result -= math.log(i)
    return result


def IMM(k, e, l):
    l = l * (1 + math.log(2) / math.log(n))
    R = Sampling(k, e, l)
    t1 = time.time()
    Sk = nodeSelection(R, k)
    t2 = time.time()
    print(t2-t1)
    print("R set: "+str(len(R)))
    return Sk


def generateRR_IC(map, rev_map, seed):
    RR = [seed]
    active = [seed]
    while len(active) != 0:
        newAck = []
        for seed in active:
            for pair in rev_map[seed]:
                neighbor, weight = pair
                if neighbor not in RR:
                    if np.random.random() <= weight:
                        RR.append(neighbor)
                        newAck.append(neighbor)
        active = newAck
    return RR


def generateRR_LT(map, rev_map, seed):
    RR = [seed]
    active = [seed]
    while len(active) != 0:
        newAck = []
        for i in active:
            for pair in rev_map[i]:
                neighbor = pair[0]
                if neighbor not in RR and getTotalWeight(map, neighbor, RR) >= np.random.random():
                    RR.append(neighbor)
                    newAck.append(neighbor)
        active = newAck
    return RR

def getTotalWeight(map, node, RR):
    thresh = 0
    for pair in map[node]:
        if pair[0] in RR:
            thresh += pair[1]
    return thresh


if __name__ == '__main__':
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default='NetHEPT.txt')
    parser.add_argument('-k', '--seed_count', type=int, default=500)
    parser.add_argument('-m', '--model', type=str, default='LT')
    parser.add_argument('-t', '--time_limit', type=int, default=60)

    args = parser.parse_args()
    file_name = os.path.abspath(args.file_name)
    seed_count = args.seed_count
    model = args.model
    time_limit = args.time_limit
    map = {}
    rev_map = {}

    n, v = readNet(file_name)
    global func
    if model == 'LT':
        func = generateRR_LT
    else:
        func = generateRR_IC

    createProcess()


    e = 0.2  # 0.5
    l = 1  # 1

    S, k = IMM(seed_count, e, l)
    close()

    # for i in S:
    #     print(i)

    print(time.time()-t0)

    '''
    程序结束后强制退出，跳过垃圾回收时间, 如果没有这个操作会额外需要几秒程序才能完全退出
    '''
    sys.stdout.flush()

import multiprocessing as mp
import time
import sys
import argparse
import os
import numpy as np
import math

core = 2
map = {}
rev_map = {}
process_list = []


class MyProcess(mp.Process):
    def __init__(self, func):
        super(MyProcess, self).__init__(target=self.start)
        self.inQueue = mp.Queue()
        self.outQueue = mp.Queue()
        self.n = n
        self.pro_map = map
        self.pro_revmap = rev_map
        self.func = func

    def run(self):
        while True:
            loop = self.inQueue.get()
            #print("process")
            R = []
            for i in range(loop):
                seed = np.random.randint(1, self.n)
                #print(seed)
                rr = self.func(self.pro_revmap, self.pro_map, seed)
                R.append(rr)
            self.outQueue.put(R)


def createProcess():
    for i in range(core):
        process = MyProcess(func)
        process.start()
        process_list.append(process)

def close():
    for p in process_list:
        p.terminate()

def initial(dic):
    for i in range(1, n + 1):
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
    e_p = e ** 0.5
    lambda_p = ((2 + 2 * e_p / 3) * n * (log_nk(n, k) + l * math.log(n)
                                         + math.log(math.log2(n)))) / math.pow(e_p, 2)
    for i in range(1, int(math.log2(n - 1) + 1)):
        x = n / math.pow(2, i)
        theta_i = lambda_p / x
        while len(R) <= theta_i:
            # generate RR
            delta = math.ceil((theta_i - len(R)) / core)
            # R_list = generateRR(delta,model,n)
            for p in process_list:
                p.inQueue.put(delta)
            for p in process_list:
                res = p.outQueue.get()
                R += res

        si, Fr = nodeSelection(R, k)
        if n * Fr >= (1 + e_p) * x:
            LB = n * Fr / (1 + e_p)
            break
    alpha = (l * math.log(n) + math.log(2)) ** 0.5
    beta = ((1 - 1 / math.e) * (log_nk(n, k) + l * math.log(n) + math.log(2))) ** 0.5
    lambda_sta = 2 * n * math.pow(((1 - 1 / math.e) * alpha + beta), 2) * math.pow(e, -2)
    theta = lambda_sta / LB
    while len(R) <= theta:
        # generate RR
        delta = math.ceil((theta - len(R)) / core)
        for p in process_list:
            p.inQueue.put(delta)
        for p in process_list:
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
    Sk = nodeSelection(R, k)
    return Sk


# def generateRR(loop,res,model,n):
#     if model == 'LT':
#         return generateRR_LT(loop,res,n)
#     else:
#         return generateRR_IC(loop,res,n)


def generateRR_IC(rev_map, map, seed):
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


def generateRR_LT(rev_map, map, seed):
    RR = [seed]
    active = [seed]
    thresh = np.random.rand(n + 1)
    while len(active) != 0:
        newAck = []
        for i in active:
            for pair in rev_map[i]:
                neighbor = pair[0]
                if neighbor not in RR and getTotalWeight(neighbor, RR, map) >= thresh[neighbor]:
                    RR.append(neighbor)
                    newAck.append(neighbor)
        active = newAck
    return RR


def getTotalWeight(node, RR, map):
    thresh = 0
    for pair in map[node]:
        if pair[0] in RR:
            thresh += pair[1]
    return thresh


# def multipro(loop):
#     result = mp.Manager().list()
#     p_list = []
#     for i in range(core):
#         p = mp.Process(target=generateRR,args= (loop,result,model,n))
#         p.start()
#         p_list.append(p)
#     for p in p_list:
#         p.join()
#     return result


if __name__ == '__main__':
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default='network.txt')
    parser.add_argument('-k', '--seed_count', type=int, default=5)
    parser.add_argument('-m', '--model', type=str, default='IC')
    parser.add_argument('-t', '--time_limit', type=int, default=20)

    args = parser.parse_args()
    file_name = os.path.abspath(args.file_name)
    seed_count = args.seed_count
    model = args.model
    time_limit = args.time_limit

    n, v = readNet(file_name)
    if model == 'LT':
        func = generateRR_LT
    else:
        func = generateRR_IC

    createProcess()
    looptime = -1
    cnt = np.zeros(n+1,float)
    e = 0.05
    l = 4

    S, k = IMM(seed_count, e, l)


    for i in S:
        print(i)
    close()
    '''
    程序结束后强制退出，跳过垃圾回收时间, 如果没有这个操作会额外需要几秒程序才能完全退出
    '''
    sys.stdout.flush()

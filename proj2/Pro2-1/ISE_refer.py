import numpy
import time
import argparse
import random
import sys

weight = {}
adjcent = {}
active = {}
w_total = {}
theta = {}

def IC(ActivitySet):
    # print(ActivitySet)
    # print(type(ActivitySet))
    count = len(ActivitySet)
    while len(ActivitySet)!=0:
        newActivitySet = []
        for seed in ActivitySet:
            if seed not in adjcent:
                continue
            for nei in adjcent[seed]:
                if active[nei]==0 and random.random()<=weight[(seed,nei)]:
                    newActivitySet.append(nei)
                    active[nei]=1
        count+=len(newActivitySet)
        ActivitySet=newActivitySet
    return count

def LT(ActivitySet):
    for e in theta:
        theta[e] = random.random()
    count = len(ActivitySet)
    while len(ActivitySet)!=0:
        newActivitySet = []
        for seed in ActivitySet:
            if seed not in adjcent:
                continue
            for nei in adjcent[seed]:
                w_total[nei]+=weight[(seed,nei)]
                if active[nei]==0 and w_total[nei]>=theta[nei]:
                    newActivitySet.append(nei)
                    active[nei]=1
        count+=len(newActivitySet)
        ActivitySet=newActivitySet
    return count

def reset(seeds):
    for e in active:
        if e not in seeds:
            active[e]=0
def reset_w_total():
    for e in w_total:
        w_total[e]=0

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default='network.txt')
    parser.add_argument('-s', '--seed', type=str, default='network_seeds.txt')
    parser.add_argument('-m', '--model', type=str, default='LT')
    parser.add_argument('-t', '--time_limit', type=int, default=60)

    args = parser.parse_args()
    graph_file = open(args.file_name,'r',encoding='utf-8')
    seed_file = open(args.seed,'r',encoding='utf-8')
    model = args.model
    time_limit = args.time_limit

    # record the graph
    list = graph_file.readlines()
    line = list[0].strip('\n').split(' ')
    n=line[0]
    length = int(line[1])
    for i in range(1,length+1):
        line = list[i].strip('\n').split(' ')
        fr = int(line[0])
        to = int(line[1])
        we = float(line[2])
        weight[(fr,to)] = we
        active[fr]=0
        active[to]=0
        w_total[fr]=0
        w_total[to]=0
        theta[fr]=0
        theta[to]=0
        if fr not in adjcent:
            adjcent[fr]=[to]
        else:
            adjcent[fr].append(to)
    seeds=[]
    list = seed_file.read().split('\n')
    for l in list:
        num=int(l)
        seeds.append(num)
        active[num] = 1


    #process the ISE process

    sum = 0
    num = 0
    if model == 'IC':
        for i in range(10000):
            if time_limit-(time.time()-start) <5:
                break
            cnt = IC(seeds)
            sum+=cnt
            # print(cnt)
            reset(seeds)
            num+=1
    else:
        for i in range(10000):
            if time_limit - (time.time() - start) < 5:
                break
            cnt = LT(seeds)
            sum+=cnt
            # print(cnt)
            reset(seeds)
            reset_w_total()
            num+=1
    print(sum/num)
    sys.stdout.flush()

# for e in adjcent:
#     print(str(e)+":"+str(adjcent[e]))
# print("---------------------------------------")
# for e in weight:
#     print(str(e)+":"+str(weight[e]))

# print(graph_file, seed_file, model, time_limit)
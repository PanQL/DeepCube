import os

import sys
import numpy as np
import argparse
import time
import json
from multiprocessing import Process, Queue
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from environments import env_utils
from ml_utils import nnet_utils
from ml_utils import search_utils

import gc

model_loc = "./savedModels/cube3/1/" # 模型所在位置，相对manage.py所在目录的路径
model_name = 'model.meta'
env = 'CUBE3'
Environment = env_utils.getEnvironment(env)


def getResult(states):
    nnet_parallel = 100
    depth_penalty = 0.2
    bfs = 0
    startIdx = 0
    endIdx = -1
    verbose = False
    useGPU = False

    if endIdx == -1:
        endIdx = len(states)

    states = states[startIdx:endIdx]
    print('state:',states)

    ### Load nnet
    from ml_utils import nnet_utils
    from ml_utils import search_utils

    if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
        gpuNums = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(",")]
    else:
        gpuNums = [None]
    numParallel = len(gpuNums)

    ### Initialize files
    dataQueues = []
    resQueues = []
    for num in range(numParallel):
        dataQueues.append(Queue(1))
        resQueues.append(Queue(1))

        dataListenerProc = Process(target=dataListener, args=(dataQueues[num],resQueues[num],gpuNums[num],))
        dataListenerProc.daemon = True
        dataListenerProc.start()


    def heuristicFn_nnet(x):
        ### Write data
        parallelNums = range(min(numParallel,x.shape[0]))
        splitIdxs = np.array_split(np.arange(x.shape[0]),len(parallelNums))
        for num in parallelNums:
            dataQueues[num].put(x[splitIdxs[num]])

        ### Check until all data is obtaied
        results = [None]*len(parallelNums)
        for num in parallelNums:
            results[num] = resQueues[num].get()

        results = np.concatenate(results)

        return(results)

    ### Get solutions
    BestFS_solve = search_utils.BestFS_solve([states],heuristicFn_nnet,Environment,bfs=bfs)
    isSolved, solveSteps, nodesGenerated_num = BestFS_solve.run(numParallel=nnet_parallel,depthPenalty=depth_penalty,verbose=verbose)
    BestFS_solve = []

    del BestFS_solve
    gc.collect()

    soln = solveSteps[0]
    return soln

# 检查求解得到的动作序列是否正确
def validSoln(state,soln,Environment):
    solnState = state
    for move in soln:
        solnState = Environment.next_state(solnState,move)

    return(Environment.checkSolved(solnState))

def dataListener(dataQueue,resQueue,gpuNum=None):
    # 导入神经网络模型
    nnet = nnet_utils.loadNnet(model_loc,model_name,False,Environment,gpuNum=gpuNum)
    while True:
        data = dataQueue.get()
        nnetResult = nnet(data)
        resQueue.put(nnetResult)


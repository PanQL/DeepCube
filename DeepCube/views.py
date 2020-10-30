from django.http import HttpResponse
from django.shortcuts import render
import json
import os
import sys
from pathlib import Path
from . import tools


def initF(request):
    return render(request, 'mofang.html')

def stateInit(request):
     if request.method=='POST':
         with open('initState.json', 'r') as f:
             result = json.load(f)
             return HttpResponse(json.dumps(result),content_type="application/json") 

def solveCube(request):
    if request.POST:
        data = request.POST.getlist("state")
        data = json.loads(data[0])
        data = tools.getResult(data['states'])
        # 开始调用模型
        solveMoves = []
        solveMoves_rev = []
        solution_text = []

        for i in data:
            solveMoves.append(i[0] + "_" + str(i[1]))
            solveMoves_rev.append(i[0] + "_" + str(-i[1]))
            if i[1] == 1:
                solution_text.append(i[0])
            else:
                solution_text.append(str(i[0]) + "\'")

        data = {"moves": solveMoves, "moves_rev": solveMoves_rev, "solve_text": solution_text}
        data = json.dumps(data)
        return HttpResponse(data)
    return render(request, 'mofang.html')


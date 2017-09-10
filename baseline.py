# processing time shoule be considered: processing is k for each hit
from math import log
from EM import *
import json
import random


def entropy(drb):
    eps = 1e-7
    ans = 0
    for x in drb:
        ans = ans - x * log(x+eps)
    return ans


def get_available_workers(num_of_workers, assignment):
    ava_workers = range(num_of_workers)
    if len(assignment) == 0:
        return ava_workers
    for ass_index in range(len(assignment)):
        for ass_key in assignment[ass_index]:
            if len(assignment[ass_index][ass_key]) != 0:
                ava_workers.remove(ass_key)
    return ava_workers


def update_available_worker(assignment):

    if len(assignment) != 0:
        print assignment
        for index in range(len(assignment)):
            for key in assignment[index]:
                if len(assignment[index][key]) != 0: # check
                    assignment[index][key].pop() # check minus one task

    print assignment
    print ""


def check_open_tasks(worker, num_of_tasks, assign_tbw):
    tasks = range(num_of_tasks)
    for task in range(num_of_tasks):
        if assign_tbw[task][worker] == 1:
            tasks.remove(task)
    return tasks


def update_assign_tbw(worker, assigned_tasks, assign_tbw, processing):
    exist = False
    for ass in processing:
        for key in ass:
            if key == worker:
                exist = True
                ass[key] = assigned_tasks

    if exist is False:
        ass = {worker: assigned_tasks}
        processing.append(ass)


def assign(eval, num_of_tasks, available_workers, assign_tbw, quality, tasks, k, processing):
    for worker in available_workers:
        open_tasks = check_open_tasks(worker, num_of_tasks, assign_tbw) #consider tasks being processed?
        if eval == "accuracy":
            assigned_task_dist, assigned_tasks = assign_accuracy(quality[worker], tasks, k, open_tasks)  # tasks should be available to this worker
        elif eval == "fscore":
            assigned_task_dist, assigned_tasks = assign_fscore(quality[worker], tasks, k, open_tasks)  # tasks should be available to this worker
        update_assign_tbw(worker, assigned_tasks, assign_tbw, processing)
    print ""
    # update_Qc()
    # update_Qw()


def inference(completed_tasks):
    pass


def assign_accuracy(worker_quality, tasks, k, open_tasks): #tasks should be available to this worker
    quality = worker_quality
    quesNum = k
    etpList = []
    # print "assign accuracy, tasks: ", tasks
    for x in range(len(tasks)):
        if x in open_tasks:
            dis = tasks[x]
            etp = entropy(dis)
            l = len(dis) # number of labels
            for t in range(l):
                delta = dis[t] * quality + (1.0 - dis[t]) * (1 - quality) / (l - 1)
                disTemp = []
                for j in range(l):
                    if (t != j):
                        disTemp.append(dis[j] * (1.0 - quality) / (l - 1) / delta)
                    else:
                        disTemp.append(dis[j] * quality / delta)
                etp = etp - entropy(disTemp) * delta
            etpList.append((x, etp))
    etpList = sorted(etpList, key=lambda x: x[1], reverse=True)
    res = [tasks[etpList[x][0]] for x in range(quesNum)] # worker estimated distribution
    assigned_tasks = [etpList[x][0] for x in range(quesNum)]
    return res, assigned_tasks


def assign_fscore(worker_quality, tasks, k, open_tasks):
    quality = worker_quality
    quesNum = k
    etpList = []
    for x in range(len(tasks)):
        if x in open_tasks:
            dis = json.loads(tasks[x].distribution)
            etp = entropy(dis)
            l = len(dis)
            sample = random.random()
            for t in range(l):
                delta = dis[t] * quality + (1.0 - dis[t]) *(1-quality)/(l-1)
                sample = sample - dis[t]
                disTemp =[]
                if (sample < 1e-7):
                    for j in range(l):
                        if (t!=j):
                            disTemp.append(dis[j]*(1.0-quality)/(l-1)/delta)
                        else:
                            disTemp.append(dis[j]*quality/delta)
                    etp = etp- entropy(disTemp)
                    break
            etpList.append((x,etp))
    etpList = sorted(etpList, key=lambda x: x[1], reverse=True)
    res = [tasks[etpList[x][0]] for x in range(quesNum)]
    assigned_tasks = [etpList[x][0] for x in range(quesNum)]
    return res, assigned_tasks


#update Qc
def update_Qc(quality, worker_answer, task, Qc):
    dis = json.loads(task.distribution)

    delta = dis[worker_answer] * quality + (1.0 - dis[worker_answer]) * (1.0 - quality) / (len(dis) - 1)
    for i in range(len(dis)):
        if i != worker_answer:
            dis[i] = dis[i] * (1.0 - quality) / (len(dis) - 1) / delta
        else:
            dis[i] = dis[i] * quality / delta
    task.distribution = json.dumps(dis)


#update Qw
def update_Qw(answers, worker, tasks):
    res = answers
    resList = []
    workerQualityDict = {}
    for x in res:
        resList.append([x.taskId, x.workerId, x.result])
        workerQualityDict[x.workerId] = WorkerInfo.objects.get(workerId=x.workerId).quality

    res = EM.infer(resList, workerQualityDict)
    for x in res:
        task = TaskInfo.objects.get(taskId=x)
        task.result = res[x]
        task.save()
    for workerId in workerQualityDict:
        worker = WorkerInfo.objects.get(workerId=workerId)
        worker.quality = workerQualityDict[workerId]
        worker.save()


def qasca(num_of_workers, num_of_tasks, num_of_choices, worker_quality, task_, confidence_init, threshold):
    num_of_tasks



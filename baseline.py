# processing time shoule be considered: processing is k for each hit
from math import log
from truthfinder import *
import random

def get_result(num_of_tasks, em):
    res = em[0]
    infer_truths = np.zeros(num_of_tasks)
    for index in range(len(res)):
        infer_truths[index] = res[index] + 1
    return infer_truths


def get_infer_truths(num_of_tasks, infer_confidence):
    truths = np.zeros(num_of_tasks)
    for task in range(num_of_tasks):
        if infer_confidence[0][task] < infer_confidence[1][task]:
            truths[task] = 1
    return truths


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


def check_completed(num_of_tasks, repeats, assignment):
    for ass_index in range(len(assignment)):
        for ass_key in assignment[ass_index]:
            if len(assignment[ass_index][ass_key]) != 0:
                return False

    for i in range(num_of_tasks):
        if repeats[i] > 0:
            return False
    return True


def worker_submit_answers(assignment, tasks, quality, assign_scheme_tbw, truths, resList): # update prococessing
    if len(assignment) != 0:
        # print assignment
        for index in range(len(assignment)):
            for key in assignment[index]:
                if len(assignment[index][key]) != 0: # check
                    task = assignment[index][key].pop() # check minus one task
                    task_dist = tasks[task]
                    qua = quality[key]
                    answer = int(choices_generator(qua, len(task_dist), truths[task]))-1
                    assign_scheme_tbw[answer][task][key] = 1
                    resList.append([task, key, answer])
                    delta = task_dist[answer] * qua + (1.0 - task_dist[answer]) * (1.0 - qua) / (len(task_dist) - 1)
                    for i in range(len(task_dist)):
                        if i != answer:
                            task_dist[i] = task_dist[i] * (1.0 - qua) / (len(task_dist) - 1) / delta
                        else:
                            task_dist[i] = task_dist[i] * qua / delta
                    tasks[task] = task_dist
    # print assignment
    # print ""



def check_open_tasks(worker, num_of_tasks, assign_tbw, repeats):
    tasks = range(num_of_tasks)
    for task in range(num_of_tasks):
        if assign_tbw[task][worker] == 1 or repeats[task] <= 0:
            tasks.remove(task)
    return tasks


def update_assign_tbw(worker, assigned_tasks, assign_tbw, processing, repeats):
    exist = False
    for ass in processing:
        for key in ass:
            if key == worker:
                exist = True
                ass[key] = assigned_tasks

    if exist is False:
        ass = {worker: assigned_tasks}
        processing.append(ass)

    for task in assigned_tasks:
        repeats[task] -= 1

    # print assign_tbw


def assign(eval, num_of_tasks, available_workers, assign_tbw, quality, tasks, k, processing, repeats):
    for worker in available_workers:
        open_tasks = check_open_tasks(worker, num_of_tasks, assign_tbw, repeats) #consider tasks being processed?
        if eval == "accuracy":
            assigned_task_dist, assigned_tasks = assign_accuracy(quality[worker], tasks, k, open_tasks)  # tasks should be available to this worker
            print "assign tasks", assigned_tasks, " to worker ", worker
        elif eval == "fscore":
            assigned_task_dist, assigned_tasks = assign_fscore(quality[worker], tasks, k, open_tasks)  # tasks should be available to this worker
        update_assign_tbw(worker, assigned_tasks, assign_tbw, processing, repeats)

    # update_Qc()
    # update_Qw()


def inference(completed_tasks):
    pass


def assign_accuracy(worker_quality, tasks, k, open_tasks): #tasks should be available to this worker
    quality = worker_quality
    quesNum = k
    etpList = []
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
    if quesNum > len(etpList):
        res = [tasks[etpList[x][0]] for x in range(len(etpList))]
        assigned_tasks = [etpList[x][0] for x in range(len(etpList))]
    else:
        res = [tasks[etpList[x][0]] for x in range(quesNum)] # worker estimated distribution
        assigned_tasks = [etpList[x][0] for x in range(quesNum)]
    return res, assigned_tasks


def assign_fscore(worker_quality, tasks, k, open_tasks):
    quality = worker_quality
    quesNum = k
    etpList = []
    for x in range(len(tasks)):
        if x in open_tasks:
            dis = tasks[x]
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
    if quesNum > len(etpList):
        res = [tasks[etpList[x][0]] for x in range(len(etpList))]
        assigned_tasks = [etpList[x][0] for x in range(len(etpList))]
    else:
        res = [tasks[etpList[x][0]] for x in range(quesNum)]
        assigned_tasks = [etpList[x][0] for x in range(quesNum)]
    return res, assigned_tasks


#update Qc
# def update_Qc(quality, worker_answer, task, Qc):
#     dis = json.loads(task.distribution)
#
#     delta = dis[worker_answer] * quality + (1.0 - dis[worker_answer]) * (1.0 - quality) / (len(dis) - 1)
#     for i in range(len(dis)):
#         if i != worker_answer:
#             dis[i] = dis[i] * (1.0 - quality) / (len(dis) - 1) / delta
#         else:
#             dis[i] = dis[i] * quality / delta
#     task.distribution = json.dumps(dis)


#update Qw
# def update_Qw(answers, worker, tasks):
#     res = answers
#     resList = []
#     workerQualityDict = {}
#     for x in res:
#         resList.append([x.taskId, x.workerId, x.result])
#         workerQualityDict[x.workerId] = WorkerInfo.objects.get(workerId=x.workerId).quality
#
#     res = EM.infer(resList, workerQualityDict)
#     for x in res:
#         task = TaskInfo.objects.get(taskId=x)
#         task.result = res[x]
#         task.save()
#     for workerId in workerQualityDict:
#         worker = WorkerInfo.objects.get(workerId=workerId)
#         worker.quality = workerQualityDict[workerId]
#         worker.save()


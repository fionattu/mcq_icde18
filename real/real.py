from process_data import *
from random import randint
import time
import baseline
import numpy as np


#todo: how to make sure existing timestamps can finish all tasks


def get_ans_and_truths(dataset):
    truths = read_truths(dataset)
    workers, ans = read_worker_label(dataset)

    return workers,len(workers), truths, len(truths), ans


def get_arrival_times(dataset):
    return read_arrival_times(dataset)


def worker_arrivals_match(workers, arrivals, mode): #todo: select top active wrokers
    res = []
    if mode == "random":
        for worker in workers:
            random = randint(0,len(arrivals))
            res.append({worker:arrivals[random].items()[0][1]})
    if mode == "descend":
        pass
    return res


def sort_arrivals(worker_arrivals):
    for worker_arrival in worker_arrivals:
        arrival_list = worker_arrival.items()[0][1]
        arrival_list.sort()


def convert_to_epoch(datetime_str):
    patterns = ('%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M')
    for pattern in patterns:
        try:
            return int(time.mktime(time.strptime(str(datetime_str), pattern)))
        except ValueError:
            pass


def get_start_and_end(worker_arrivals):
    start = int(time.time())
    end = 0
    for worker_arrival in worker_arrivals:
        arrival_list = worker_arrival.items()[0][1]
        temp_start = convert_to_epoch(arrival_list[0])
        temp_end = convert_to_epoch(arrival_list[len(arrival_list) - 1])
        if temp_start < start:
            start = temp_start
        if temp_end > end:
            end = temp_end
    return start, end


def convert_to_batch(arrival, start, batch_interval):
    return (int)((int)(arrival-start)/batch_interval) + 1


def add_to_batch(worker, batch, batch_arrivals):
    for index in range(len(batch_arrivals)):
        for key in batch_arrivals[index]:
            if key == batch:
                if worker in batch_arrivals[index][key]:
                    return
                else:
                    batch_arrivals[index][key].append(worker)
                    return
    batch_arrivals.append({batch: [worker]})


def even_divider(worker_arrivals, num_of_batches):
    start, end = get_start_and_end(worker_arrivals)
    batch_interval = (float)(end - start)/num_of_batches
    batch_arrivals = []

    for worker_arrival in worker_arrivals:
        worker = worker_arrival.items()[0][0]
        arrival_list = worker_arrival.items()[0][1]
        for arrival in arrival_list:
            batch = convert_to_batch(convert_to_epoch(arrival), start, batch_interval)
            add_to_batch(worker, batch, batch_arrivals)
    return batch_arrivals


def two_end_divider(worker_arrivals, num_of_batches):
    start, end = get_start_and_end(worker_arrivals)
    batch_interval = (float)(end - start) / num_of_batches
    batch_arrivals = []

    for worker_arrival in worker_arrivals:
        worker = worker_arrival.items()[0][0]
        arriving = convert_to_epoch(worker_arrival.items()[0][1][0])
        leaving = convert_to_epoch(worker_arrival.items()[0][1][len(worker_arrival.items()[0][1]) - 1])
        arriving_batch = convert_to_batch(arriving, start,batch_interval)
        leaving_batch = convert_to_batch(leaving, start, batch_interval)
        for batch in range(arriving_batch, leaving_batch + 1):
            add_to_batch(worker, batch, batch_arrivals)
    return batch_arrivals


def divide_timestamps(worker_arrivals, num_of_batches): # 1. only use arriving and leaving times(not reasonable) 2. use all arriving times
    sort_arrivals(worker_arrivals)
    even_timestamps = even_divider(worker_arrivals, num_of_batches)
    start_and_end_timestamps = two_end_divider(worker_arrivals, num_of_batches)
    return even_timestamps, start_and_end_timestamps


def batch_mapping(timestamps, workers): #have to check
    for ts in timestamps:
        worker_list = ts.items()[0][1]
        for i in range(len(worker_list)):
            worker_list[i] = workers.index(worker_list[i])


def task_mapping(truths):
    tasks = []
    truths_list = []
    for truth in truths:
        print truth
        tasks.append(truth['task'])
        truths_list.append(truth['truth'])

    return tasks, truths_list


def ans_mapping(ans, workers, task_list):
    for answer in ans:
        answer['worker'] = workers.index(answer['worker'])
        answer['task'] = task_list.index(answer['task'])


def reorder_wid_tid(even_timestamps, start_and_end_timestamps, workers, truths, ans):
    batch_mapping(even_timestamps, workers)
    batch_mapping(start_and_end_timestamps, workers)
    task_list, truth_list = task_mapping(truths)
    ans_mapping(ans, workers, task_list)



def batch_assignment(ans_dataset, arrival_dataset, matching_mode,num_of_batches):
    workers, num_of_workers, truths, num_of_tasks, ans = get_ans_and_truths(ans_dataset)
    arrivals = get_arrival_times(arrival_dataset)
    worker_arrivals = worker_arrivals_match(workers,arrivals,matching_mode)
    even_timestamps, start_and_end_timestamps = divide_timestamps(worker_arrivals, num_of_batches)
    reorder_wid_tid(even_timestamps, start_and_end_timestamps, workers, truths, ans)
    return even_timestamps, start_and_end_timestamps, workers, truths, ans


def main(ans_dataset='d_Duck Identification_40w217q', arrival_dataset='Relevance_of_terms_to_disaster_relief_topics',matching_mode='random',num_of_batches=40):
    even_timestamps, start_and_end_timestamps, workers, truths, ans = batch_assignment(ans_dataset, arrival_dataset, matching_mode, num_of_batches)
    baseline(even_timestamps, workers, truths, ans, num_of_batches)
    # baseline(even_timestamps, start_and_end_timestamps, num_of_batches, )
    # ff(even_timestamps, start_and_end_timestamps)
    # bf(even_timestamps, start_and_end_timestamps)
    # random(even_timestamps, start_and_end_timestamps)
    # baseline(even_timestamps, num_of_batches, workers, truths,ans)

num_of_batches = 40
main()
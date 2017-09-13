from process_data import *
from random import randint
import time


#todo: how to make sure existing timestamps can finish all tasks


def get_ans_and_truths(dataset):
    truths = read_truths(dataset)
    workers, ans = read_worker_label(dataset)

    return workers,len(workers), truths, len(truths), ans


def get_arrival_times(dataset):
    return read_arrival_times(dataset)


def worker_arrivals_match(workers, arrivals, mode):
    res = []
    if mode == "random":
        for worker in workers:
            random = randint(0,len(arrivals))
            res.append({worker:arrivals[random].items()[0][1]})
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


def even_divider(worker_arrivals, num_of_batches):
    start, end = get_start_and_end(worker_arrivals)
    print "start: ", start
    print "end: ", end
    batch_interval = (float)(end - start)/num_of_batches
    batch_arrivals = []
    for worker_arrival in worker_arrivals:





def divide_timestamps(worker_arrivals, num_of_batches): # 1. only use arriving and leaving times(not reasonable) 2. use all arriving times
    sort_arrivals(worker_arrivals)
    even_timestamps = even_divider(worker_arrivals, num_of_batches)


def get_real_data(ans_dataset, arrival_dataset, matching_mode,num_of_batches):
    workers, num_of_workers, truths, num_of_tasks, ans = get_ans_and_truths(ans_dataset)
    arrivals = get_arrival_times(arrival_dataset)
    worker_arrivals = worker_arrivals_match(workers,arrivals,matching_mode)
    timestamps = divide_timestamps(worker_arrivals, num_of_batches)


def test():
    start = 1453980175
    end = 1455906578
    batch_time = (float)(end - start)/10
    print batch_time



test()
#'Tweets_about_the_Syrian_civil_war'
# 'Relevance_of_terms_to_disaster_relief_topics'
# get_real_data('d_Duck Identification_40w217q', 'Relevance_of_terms_to_disaster_relief_topics', 'random')
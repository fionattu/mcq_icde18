from process_data import *
from random import randint

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

def get_real_data(ans_dataset, arrival_dataset, matching_mode):
    workers, num_of_workers, truths, num_of_tasks, ans = get_ans_and_truths(ans_dataset)
    arrivals = get_arrival_times(arrival_dataset)
    worker_arrivals = worker_arrivals_match(workers,arrivals,matching_mode)


def test():
    dict = {1:[2,3]}
    print dict.items()[0][1]

#'Tweets_about_the_Syrian_civil_war'
# 'Relevance_of_terms_to_disaster_relief_topics'
get_real_data('d_Duck Identification_40w217q', 'Relevance_of_terms_to_disaster_relief_topics', 'random')
import datetime
import math
from real import *
from mcq_inference import *
import logging


def check_assignment(worker, task, assign_scheme_tbw):
    if assign_scheme_tbw[0][task][worker] == 1 or assign_scheme_tbw[1][task][worker] == 1:
        return True
    else:
        return False


def get_answer(worker, task, ans):
    return find_answer(worker,task,ans)


def prescan(num_of_workers, num_of_tasks, num_of_choices, workers, ans, task_capacity, dei_wbt, assign_scheme_tbw, completed_tasks, truths, expertise_truths, difficulty_truths): # label completed tasks
    # a question is assigned if all workers can complete it: ok(largest negative remaining capacity over workers)
    # a question is assigned if there is a worker who can complete it
    remain_capacity_wbt = np.subtract(task_capacity, dei_wbt)
    available_workers = workers
    for i in range(num_of_tasks):
        if i not in completed_tasks:
            min = 1000
            min_worker = -1
            for j in range(num_of_workers):
                if j in available_workers:
                    current = remain_capacity_wbt[j][i]
                    if current <= 0 and check_assignment(j, i, assign_scheme_tbw) is not True:
                        if min > 0 or (min < 0 and min < current): #have to check
                            min = current
                            min_worker = j
            if min <= 0:        #assign task i to min_worker
                choice = get_answer(min_worker, i, ans)
                assign_scheme_tbw[choice][i][min_worker] = 1
                print "prescan assign worker ", min_worker, " task ", i
                available_workers.remove(min_worker)
                # print "prescan remove worker", min_worker
                completed_tasks.append(i)

                logging.info("prescan assign worker %d minworker %d", min_worker, i)
    return available_workers


def assign_first_open(num_of_tasks, num_of_choices, ans, dei_wbt, remain_capacity_wbt, assign_scheme_tbw, completed_tasks, available_workers, open_tasks, truths, expertise_truths, difficulty_truths):
    # print "enter assign first open"
    min_task = ""
    min_worker = ""
    min_dei = 10000
    for w in available_workers:
        for t in range(num_of_tasks):
            if check_assignment(w,t,assign_scheme_tbw) is not True:
                current = remain_capacity_wbt[w][t]
                if t not in completed_tasks and current < min_dei:
                    min_dei = current
                    min_worker = w
                    min_task = t

    if min_task != "":
        process_assignment(min_worker, min_task, num_of_choices, ans, assign_scheme_tbw, available_workers,dei_wbt,remain_capacity_wbt,truths, expertise_truths, difficulty_truths)
        open_tasks.append(min_task)
        # print "exit assign first open"
        return True
    else:
        return False


def update_dei(worker, task, reduced_dei, remain_capacity_wbt):
    remain_capacity_wbt[:,task] = remain_capacity_wbt[:,task] - [reduced_dei] #have to check
    remain_capacity_wbt[worker, task] += reduced_dei


def process_assignment(worker, task, num_of_choices, ans, assign_scheme_tbw, available_workers, dei_wbt, remain_capacity_wbt, truths, expertise_truths, difficulty_truths):
    choice = get_answer(worker, task, ans)
    assign_scheme_tbw[choice][task][worker] = 1
    update_dei(worker, task, dei_wbt[worker][task], remain_capacity_wbt)
    print "process assign worker ", worker, " task ", task
    available_workers.remove(worker)
    # print "process remove worker:", worker

    logging.info("process assign worker %d task %d", worker, task)


def assign_to_first_open(ans, worker, dei_wbt, open_tasks, assign_scheme_tbw, available_workers,remain_capacity_wbt, num_of_choices, truths, expertise_truths, difficulty_truths):
    for task in open_tasks:
        if remain_capacity_wbt[worker][task] >= 0 and check_assignment(worker, task, assign_scheme_tbw) is False:
            process_assignment(worker, task, num_of_choices, ans, assign_scheme_tbw, available_workers, dei_wbt, remain_capacity_wbt, truths, expertise_truths, difficulty_truths)
            return True
    return False


def assign_to_closed(ans, worker, num_of_tasks, dei_wbt, remain_capacity_wbt, assign_scheme_tbw, completed_tasks, available_workers, open_tasks, num_of_choices,truths, expertise_truths, difficulty_truths):
    min_task = ""
    min_dei = 10000
    for task in range(num_of_tasks):
        if check_assignment(worker, task, assign_scheme_tbw) is False:
            current = remain_capacity_wbt[worker][task]
            if task not in completed_tasks and 0 <= current < min_dei:
                min_task = task
                min_dei = current
    if min_task != "":
        process_assignment(worker, min_task, num_of_choices, ans, assign_scheme_tbw, available_workers, dei_wbt, remain_capacity_wbt, truths, expertise_truths, difficulty_truths)
        open_tasks.append(min_task)
        return True
    return False # worker has completed all closed tasks in former iterations


def start_first_fit(num_of_workers, num_of_tasks, num_of_choices, ans, task_capacity, dei_wbt, assign_scheme_tbw, completed_tasks, available_workers, truths, expertise_truths, difficulty_truths):
    remain_capacity_wbt = np.subtract(task_capacity, dei_wbt) # after prescan, available_worker have "positive" dei over all non-completed tasks
    open_tasks = []
    #still have incompleted tasks, but available workers have all done them, has to wait for new workers
    next_assignment_feasible = assign_first_open(num_of_tasks, num_of_choices, ans, dei_wbt, remain_capacity_wbt, assign_scheme_tbw, completed_tasks, available_workers, open_tasks, truths, expertise_truths, difficulty_truths)
    if next_assignment_feasible is True:
        for worker in range(num_of_workers):
            if worker in available_workers:
                # print "open tasks:", open_tasks
                if assign_to_first_open(ans, worker, dei_wbt, open_tasks, assign_scheme_tbw, available_workers,remain_capacity_wbt,num_of_choices, truths, expertise_truths, difficulty_truths) is False:
                    # return false no tasks suitable (workers have have full assignment for remaining tasks(then quit) or are (not possible)left with exceeding capacity tasks(next round prescan))
                    if len(open_tasks) < num_of_tasks:
                        assign_to_closed(ans, worker, num_of_tasks, dei_wbt, remain_capacity_wbt, assign_scheme_tbw,completed_tasks, available_workers, open_tasks, num_of_choices, truths, expertise_truths,difficulty_truths)
                    else:
                        continue
                        # # no tasks suitable (haven't considered, #open task = #tasks, may assign with smallest exceeding capacity )
                        # # assign_to_random_open not in assigned tasks for this worker
                        # # random_open_task = select_random_task(worker, num_of_tasks, assign_scheme_tbw, completed_tasks)
                        # # if random_open_task != -1:
                        # process_assignment(worker, open_tasks[0], num_of_choices, assign_scheme_tbw, available_workers, dei_wbt, remain_capacity_wbt,truths, expertise_truths, difficulty_truths)

def first_fit_greedy(num_of_workers, num_of_tasks, num_of_choices, workers, ans, task_capacity, dei_wbt, assign_scheme_tbw, completed_tasks, truths, expertise_truths, difficulty_truths):
    available_workers = prescan(num_of_workers, num_of_tasks, num_of_choices, workers, ans, task_capacity, dei_wbt, assign_scheme_tbw, completed_tasks, truths, expertise_truths, difficulty_truths)
    if len(available_workers) != 0 and len(completed_tasks) < num_of_tasks:
        start_first_fit(num_of_workers, num_of_tasks, num_of_choices, ans, task_capacity, dei_wbt, assign_scheme_tbw, completed_tasks, available_workers, truths, expertise_truths, difficulty_truths)


def assign_to_best_open(ans, worker, dei_wbt, open_tasks, assign_scheme_tbw, available_workers, remain_capacity_wbt, num_of_choices, truths, expertise_truths, difficulty_truths):
    min = 1000
    min_task = -1
    for task in open_tasks:
        if check_assignment(worker, task, assign_scheme_tbw) is False:
            current = remain_capacity_wbt[worker][task]
            if current >= 0 and current < min:
                min = current
                min_task = task
    if min_task != -1:
        process_assignment(worker, task, num_of_choices, ans, assign_scheme_tbw, available_workers, dei_wbt, remain_capacity_wbt, truths, expertise_truths, difficulty_truths)
        return True
    return False


def start_best_fit(num_of_workers, num_of_tasks, num_of_choices, ans, task_capacity, dei_wbt, assign_scheme_tbw, completed_tasks, available_workers, truths, expertise_truths, difficulty_truths):
    remain_capacity_wbt = np.subtract(task_capacity,dei_wbt)  # after prescan, available_worker have "positive" dei over all non-completed tasks
    open_tasks = []
    next_assignment_feasible = assign_first_open(num_of_tasks, num_of_choices, ans,dei_wbt, remain_capacity_wbt, assign_scheme_tbw, completed_tasks, available_workers, open_tasks, truths, expertise_truths, difficulty_truths)
    if next_assignment_feasible is True:
        for worker in range(num_of_workers):
            if worker in available_workers:
                if assign_to_best_open(ans, worker, dei_wbt, open_tasks, assign_scheme_tbw, available_workers,remain_capacity_wbt, num_of_choices, truths, expertise_truths, difficulty_truths) is False:
                    # 1. open next open
                    # 2. no tasks suitable (haven't considered, #open task = #tasks, may assign with smallest exceeding capacity )
                    if len(open_tasks) < num_of_tasks:
                        assign_to_closed(ans, worker, num_of_tasks, dei_wbt, remain_capacity_wbt, assign_scheme_tbw, completed_tasks, available_workers, open_tasks, num_of_choices, truths, expertise_truths,difficulty_truths)
                    else:
                        continue# 2. no tasks suitable (haven't considered, #open task = #tasks, may assign with smallest exceeding capacity )
                        # assign_to_random_open
                        # process_assignment(worker, open_tasks[0], num_of_choices, assign_scheme_tbw, available_workers, dei_wbt, remain_capacity_wbt,truths, expertise_truths, difficulty_truths)


def best_fit_greedy(num_of_workers, num_of_tasks, num_of_choices, workers, ans, task_capacity, dei_wbt, assign_scheme_tbw, completed_tasks, truths, expertise_truths, difficulty_truths):
    available_workers = prescan(num_of_workers, num_of_tasks, num_of_choices, workers, ans, task_capacity, dei_wbt, assign_scheme_tbw,completed_tasks, truths, expertise_truths, difficulty_truths)
    print "after prescan completed: ", completed_tasks
    # logging.info("after prescan completed: %d", completed_tasks)
    if len(available_workers) != 0 and len(completed_tasks) < num_of_tasks:
        start_best_fit(num_of_workers, num_of_tasks, num_of_choices, ans, task_capacity, dei_wbt, assign_scheme_tbw, completed_tasks, available_workers, truths, expertise_truths, difficulty_truths)


def assign_with_mode(assign_mode, num_of_workers, num_of_tasks, num_of_choices, workers, ans, task_capacity, dei_wbt, assign_scheme_tbw, completed_tasks, truths, expertise_truths, difficulty_truths): # cold start
    if assign_mode is "firstfit":
        first_fit_greedy(num_of_workers, num_of_tasks, num_of_choices, workers, ans, task_capacity, dei_wbt, assign_scheme_tbw, completed_tasks, truths, expertise_truths, difficulty_truths)

    elif assign_mode is "bestfit":
        best_fit_greedy(num_of_workers, num_of_tasks, num_of_choices, workers, ans, task_capacity, dei_wbt, assign_scheme_tbw, completed_tasks, truths, expertise_truths, difficulty_truths)


def calculate_prob_ans_wbt(prob_ans_eq_truth_wbt, infer_confidence, num_of_workers, num_of_tasks, num_of_choices):
    #check element of infer_confidence should be of 1 x tasks
    prob_ans_neq_truth_wbt = (1 - prob_ans_eq_truth_wbt)/(num_of_choices - 1)
    # worker selection probability
    prob_ans_wbt = [np.zeros((num_of_workers, num_of_tasks)) for _ in range(num_of_choices)]
    prob_ans_wbt[0] = prob_ans_eq_truth_wbt * np.asarray(infer_confidence[0]) + prob_ans_neq_truth_wbt * np.asarray(infer_confidence[1])
    prob_ans_wbt[1] = prob_ans_neq_truth_wbt * np.asarray(infer_confidence[0]) + prob_ans_eq_truth_wbt * np.asarray(infer_confidence[1])

    return prob_ans_wbt


def calculate_ei(infer_confidence_score, infer_confidence, infer_expertise_score, infer_difficulty, estimated_difficulty_score, num_of_workers, num_of_tasks, num_of_choices):
    #infer_expertise_score and infer_difficulty are "1 x tasks"
    ei_wbt = [np.zeros((num_of_workers, num_of_tasks)) for _ in range(num_of_choices)]
    updated_confidence_score = [np.zeros((num_of_workers, num_of_tasks)) for _ in range(num_of_choices)]
    updated_dampen_confidence_score = [np.zeros((num_of_workers, num_of_tasks)) for _ in range(num_of_choices)]
    updated_confidence = [np.zeros((num_of_workers, num_of_tasks)) for _ in range(num_of_choices)]
    updated_difficulty_score = [np.zeros((num_of_workers, num_of_tasks)) for _ in range(num_of_choices)]
    for i in range(num_of_choices):
        updated_confidence_score[i] = np.add(np.asarray([infer_confidence_score[i],] * num_of_workers), np.asarray(infer_expertise_score)[:,None]) # wbt # have to add dampen factors
    updated_dampen_confidence_score[0] = updated_confidence_score[0] - 0.6*updated_confidence_score[1] #todo
    updated_dampen_confidence_score[1] = updated_confidence_score[1] - 0.6*updated_confidence_score[0] #todo

    for i in range(num_of_choices):
        updated_confidence[i] = (1/(1 + np.power(math.e, -5*updated_dampen_confidence_score[i]))) * np.asarray(infer_difficulty) #make sure the infer/update confidence have the same difficulties
        if i is 0:
            updated_difficulty_score[i] = np.abs(np.subtract(np.asarray(updated_confidence[i]),np.asarray(infer_confidence[1])))
        elif i is 1:
            updated_difficulty_score[i] = np.abs(np.subtract(updated_confidence[i],infer_confidence[0]))

        ei_wbt[i] = np.subtract(updated_difficulty_score[i],estimated_difficulty_score)
    return ei_wbt


def calculate_dei(num_of_workers, num_of_tasks, num_of_choices, infer_expertise, infer_expertise_score, infer_difficulty, estimated_difficulty_score, infer_confidence, infer_confidence_score):
    # infer_confidence_score is 2 x tasks, infer_expertise_score is 1 x workers
    prob = np.transpose(np.multiply(np.asarray(infer_expertise), np.asarray(infer_difficulty)[:, None]))
    prob_ans_eq_truth_wbt = 1 / (1 + np.power(math.e, -3*prob)) #todo: 3 can be tuned
    prob_ans_wbt = calculate_prob_ans_wbt(prob_ans_eq_truth_wbt, infer_confidence, num_of_workers, num_of_tasks, num_of_choices)
    ei_wbt = calculate_ei(infer_confidence_score, infer_confidence, infer_expertise_score, infer_difficulty, estimated_difficulty_score, num_of_workers, num_of_tasks, num_of_choices)
    dei_wbt = np.abs(np.add(np.multiply(prob_ans_wbt[0],ei_wbt[0]),np.multiply(prob_ans_wbt[1],ei_wbt[1]))) # should be abs
    return dei_wbt


def process_completed_tasks(num_of_tasks, threshold, infer_difficulty_score, infer_confidence, completed_tasks, infer_truths):
    for i in range(num_of_tasks):
        if i in completed_tasks and infer_truths[i] == 0:
            infer_truths[i] = 0 if infer_confidence[0][i] > infer_confidence[1][i] else 1
        if i not in completed_tasks and infer_difficulty_score[i] >= threshold:
            infer_truths[i] = 0 if infer_confidence[0][i] > infer_confidence[1][i] else 1
            completed_tasks.append(i)

    for i in range(num_of_tasks):
        print "task ", i, " has inferred difficulty score: ", infer_difficulty_score[i], " has confidence: ", infer_confidence[0][i], " and ", infer_confidence[1][i], " ", i in completed_tasks
        logging.info("task %d has infer has inferred difficulty score: %f and confidence %f and %f", i,infer_difficulty_score[i], infer_confidence[0][i], infer_confidence[1][i])


def synthetic_exp(assign_mode, timestamps, truths, worker_set, ans, num_of_batches, num_of_choices, num_of_tasks, expertise_init, difficulty_init, confidence_init, threshold):
    num_of_workers = len(worker_set)
    print "worker len: ", num_of_workers
    print "task len", num_of_tasks
    infer_expertise = np.zeros(num_of_workers)
    infer_expertise_score = [-np.log(1-expertise_init)] * num_of_workers
    infer_confidence = [[confidence_init] * num_of_tasks for _ in range(num_of_choices)]
    infer_confidence_score = [ np.zeros(num_of_tasks) for _ in range(num_of_choices)]
    infer_difficulty = [difficulty_init] * num_of_tasks
    assign_scheme_tbw = [np.zeros((num_of_tasks, num_of_workers)) for _ in range(num_of_choices)]  # assignmnet scheme
    difficulty_truths = [uniform_random_generator(0.8,1)] * num_of_tasks
    expertise_truths = [uniform_random_generator(0.5, 0.999)] * num_of_workers
    infer_truths = np.zeros(num_of_tasks)
    completed_tasks = []
    current_batch = 1
    workers = []
    while(len(completed_tasks) < num_of_tasks): #begin a batch, old workers/tasks: last batch paras, new workers/tasks: initialized

        print assign_mode, "______________________batch______________________", current_batch
        if current_batch <= num_of_batches:
            batch_workers = find_batch_workers(current_batch,timestamps)  # add avaliable worker set, worker set can be empty
            add_to_workers(workers, batch_workers)  # assume workers will remain in task pool until he leaves
        # "workers" is the current available workers
        print "worker: ", workers
        estimated_difficulty_score = np.abs(np.subtract(np.asarray(infer_confidence[0]), np.asarray(infer_confidence[1])))
        task_capacity = threshold - estimated_difficulty_score #threshold can be vector or scala

        dei_wbt = calculate_dei(num_of_workers, num_of_tasks, num_of_choices, infer_expertise, infer_expertise_score,
                                                infer_difficulty, estimated_difficulty_score, infer_confidence, infer_confidence_score)

        # task processing: one task for a batch
        assign_with_mode(assign_mode, num_of_workers, num_of_tasks, num_of_choices, workers, ans, task_capacity, dei_wbt, assign_scheme_tbw, completed_tasks, truths, expertise_truths, difficulty_truths) # assign_scheme_tbw includes the answers
        [infer_expertise, infer_expertise_score, infer_confidence, infer_confidence_score, infer_difficulty, infer_difficulty_score] = start_inference(num_of_workers, num_of_tasks, num_of_choices, assign_scheme_tbw,expertise_init, difficulty_init)

        logging.info("_________________________iteration: %d_________________________", current_batch)
        process_completed_tasks(num_of_tasks, threshold, infer_difficulty_score, infer_confidence, completed_tasks, infer_truths)  # check whether completed tasks are updated
        current_batch += 1
        print "_________________________completes after assignments:", completed_tasks
        print "_________________________#total completes: ", len(completed_tasks)
        logging.info("_________________________#total completes: %d", len(completed_tasks))

    return [print_accuracy(num_of_tasks, truths, infer_truths), current_batch]


# num_of_tasks = [50,100,150,200,500,1000]
num_of_tasks = 39
iteration = 1
threshold = 0.3
num_of_batches=40

ans_dataset='d_Duck Identification_40w217q'
arrival_dataset='Relevance_of_terms_to_disaster_relief_topics'

# ans_dataset='test'
# arrival_dataset='test'
# matching_mode='random'
matching_mode='order'


num_of_choices = 2
expertise_init = 0.5
difficulty_init = 0.5
confidence_init = 0.5
max_number_of_workers = 50
accuracy_ff = 0
time_ff = 0


def print_result(assign_mode):
    logging.basicConfig(filename='./log/' + assign_mode + str(datetime.date.today().strftime("%d%m%y")) +'.log', filemode='w',level=logging.DEBUG)
    start_and_end_timestamps, workers, truths, ans = batch_assignment(ans_dataset, arrival_dataset, matching_mode,num_of_batches)

    accuracy_ff = 0
    time_ff = 0
    for ite in range(iteration):
        [accuracy, time] =synthetic_exp(assign_mode, start_and_end_timestamps, truths, workers, ans, num_of_batches, num_of_choices, num_of_tasks, expertise_init, difficulty_init, confidence_init, threshold)
        accuracy_ff += accuracy
        time_ff += time
    print assign_mode, ": "
    print " accuracy: ", 100*float(accuracy_ff)/(iteration * num_of_tasks), "%"
    print " time: ", float(time_ff)/iteration
    logging.info("%s: ", assign_mode)
    logging.info(" accuracy: %f", 100*float(accuracy_ff)/(iteration * num_of_tasks))
    logging.info(" time: %f", float(time_ff)/iteration)



# print_result('firstfit')
print_result('bestfit')
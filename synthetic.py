from truthfinder import *
import numpy as np


# worker selection probability is different from the truthfinder similation
#start assignment
    # calculate_dei(confidence of answer/expertise)
    # assign_with_mode
    #generate and store answers
    #normalization
#end assignment

#to do:
# 1. check completed tasks (pre-scan, stop assign them to workers)
# 2. cold start workers/tasks, old workers/tasks: last batch paras, new workers/tasks: initialized
# 3. consider task processing time, construct a available worker set
# 4. check workers that have done all tasks(can be implemented with 3)
# 5. normalization of confidence

def init(num_of_workers, expertise_init, num_of_tasks, difficulty_init):
    return initialization(num_of_workers, expertise_init, num_of_tasks, difficulty_init)


def prescan(task_capacity, dei_wbt, assign_scheme_tbw): # label completed tasks
    pass


def check_assignment(worker, task, assign_scheme_tbw):
    if assign_scheme_tbw[0][task][worker] is 1 or assign_scheme_tbw[1][task][worker] is 1:
        return True
    else:
        return False


def full_assign(worker, num_of_tasks, assign_scheme_tbw,completed_tasks):
    assigned_task = completed_tasks
    num_of_assign = len(assigned_task)
    for i in range(num_of_tasks):
        if i not in assigned_task and check_assignment(worker, i, assign_scheme_tbw) is True:
            assigned_task.append(i)
            num_of_assign += 1
    if num_of_assign == num_of_tasks:
        return [True, assigned_task]
    else:
        return [False, assigned_task]


def generate_random_task(num_of_tasks, completed_tasks, assigned_tasks):
    worker_ban_task_list = completed_tasks + assigned_tasks
    # print "worker ban list: ", worker_ban_task_list
    a = np.random.choice([i for i in range(0, num_of_tasks) if i not in worker_ban_task_list])
    # print "gen:", a
    return np.random.choice([i for i in range(0, num_of_tasks) if i not in worker_ban_task_list])


def select_task(worker, num_of_tasks, assign_scheme_tbw, completed_tasks):
    # print "current completed tasks: ", completed_tasks
    [full, assigned_tasks] = full_assign(worker, num_of_tasks, assign_scheme_tbw,completed_tasks)
    if full is True:
        return -1
    else:
        task = generate_random_task(num_of_tasks, completed_tasks, assigned_tasks)
        return task


def generate_answer(worker, task, prob_ans_wbt): # check normalization is needed?
    if uniform_random_generator(0,1) < prob_ans_wbt[0][worker][task]:
        return 0
    else:
        return 1


def random_assign(num_of_workers, num_of_tasks, dei_wbt, prob_ans_wbt, assign_scheme_tbw, completed_tasks): #available worker set
    for i in range(num_of_workers):
        # print "assigning worker:", i
        task = select_task(i, num_of_tasks, assign_scheme_tbw, completed_tasks)
        # print "assign worker:", i, " task ", task
        if task is not -1:
            choice = generate_answer(i, task, prob_ans_wbt) # check whether assignment_scheme_tbw is updated
            assign_scheme_tbw[choice][task][i] = 1
        else:
            pass #this worker has completed all avialable tasks, do not assign any task


def assign_with_mode(assign_mode, num_of_workers, num_of_tasks, task_capacity, dei_wbt, prob_ans_wbt, assign_scheme_tbw, completed_tasks): # cold start
    if assign_mode is "random":
        random_assign(num_of_workers, num_of_tasks, dei_wbt, prob_ans_wbt, assign_scheme_tbw, completed_tasks)
        # print assign_scheme_tbw

    elif assign_mode is "baseline":
        pass

    elif assign_mode is "firstfit":
        prescan(task_capacity, dei_wbt, assign_scheme_tbw)
        pass
    elif assign_mode is "bestfit":
        prescan(task_capacity, dei_wbt, assign_scheme_tbw)
        pass


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
    updated_confidence = [np.zeros((num_of_workers, num_of_tasks)) for _ in range(num_of_choices)]
    updated_difficulty_score = [np.zeros((num_of_workers, num_of_tasks)) for _ in range(num_of_choices)]
    for i in range(num_of_choices):
        updated_confidence_score[i] = np.add(np.asarray([infer_confidence_score[i],] * num_of_workers), np.asarray(infer_expertise_score)[:,None]) # wbt # have to add dampen factors
        updated_confidence[i] = (1/(1 + np.power(math.e, -updated_confidence_score[i]))) * np.asarray(infer_difficulty) #make sure the infer/update confidence have the same difficulties
        if i is 0:
            a=  np.asarray(updated_confidence[i]) - np.asarray(infer_confidence[1])
            updated_difficulty_score[i] = np.abs(np.subtract(np.asarray(updated_confidence[i]),np.asarray(infer_confidence[1])))
        elif i is 1:
            updated_difficulty_score[i] = np.abs(np.subtract(updated_confidence[i],infer_confidence[0]))

        ei_wbt[i] = np.subtract(updated_difficulty_score[i],estimated_difficulty_score)
    return ei_wbt


def calculate_dei(num_of_workers, num_of_tasks, num_of_choices, infer_expertise, infer_expertise_score, infer_difficulty, estimated_difficulty_score, infer_confidence, infer_confidence_score):
    # infer_confidence_score is 2 x tasks, infer_expertise_score is 1 x workers
    prob = np.transpose(np.multiply(np.asarray(infer_expertise), np.asarray(infer_difficulty)[:, None]))
    prob_ans_eq_truth_wbt = 1 / (1 + np.power(math.e, -3 * prob))
    prob_ans_wbt = calculate_prob_ans_wbt(prob_ans_eq_truth_wbt, infer_confidence, num_of_workers, num_of_tasks, num_of_choices)
    ei_wbt = calculate_ei(infer_confidence_score, infer_confidence, infer_expertise_score, infer_difficulty, estimated_difficulty_score, num_of_workers, num_of_tasks, num_of_choices)
    dei_wbt = np.abs(np.add(np.multiply(prob_ans_wbt[0],ei_wbt[0]),np.multiply(prob_ans_wbt[1],ei_wbt[1]))) # should be abs
    return [dei_wbt, prob_ans_wbt]


def start_inference(num_of_workers, num_of_tasks, num_of_choices, assign_scheme_tbw, expertise_init, difficulty_init):
    # return model paras

    # start learning
    [expertise, difficulty] = init(num_of_workers, expertise_init, num_of_tasks, difficulty_init)
    for iter_all in range(5): # whole process
        for iter_left in range(5): # left cycle
            # expertise should be value of last iteration
            expertise_score = [-np.log(1 - x) for x in expertise]
            confidence_score = [np.dot(x, expertise_score) for x in assign_scheme_tbw]
            confidence_score_damping = np.zeros((num_of_choices, num_of_tasks))
            confidence = np.zeros((num_of_choices, num_of_tasks))

            confidence_score_damping[0][:] = confidence_score[0][:] - 1 * confidence_score[1][:]
            confidence_score_damping[1][:] = confidence_score[1][:] - 1 * confidence_score[0][:]

            for task in range(num_of_tasks):
                for choice in range(num_of_choices):
                    confidence[choice][task] = difficulty[task] * (1 / (1 + np.power(math.e, -confidence_score_damping[choice][task])))
                    # confidence[choice][task] = 1 / (1 + np.power(math.e, -confidence_score_damping[choice][task]))

            for x in range(num_of_workers):
                expertise[x] = 0
                for task in range(num_of_tasks):
                    for choice in range(num_of_choices):
                        expertise[x] += assign_scheme_tbw[choice][task][x] * confidence[choice][task]
            expertise = [x / num_of_tasks for x in expertise]

        difficulty_score = np.zeros(num_of_tasks)
        for iter_right in range(5):  # right cycle
            for task in range(num_of_tasks):
                difficulty_score[task] = np.abs(confidence[0][task] - confidence[1][task])
                difficulty[task] = 1 / (1 + 0.4 * np.power(math.e, -difficulty_score[task]))
            # update confidence
            for task in range(num_of_tasks):
                for choice in range(num_of_choices):
                    confidence[choice][task] = difficulty[task] * (1 / (1 + np.power(math.e, -confidence_score_damping[choice][task])))

        # update expertise
        for x in range(num_of_workers):
            expertise[x] = 0
            for task in range(num_of_tasks):
                for choice in range(num_of_choices):
                    expertise[x] += assign_scheme_tbw[choice][task][x] * confidence[choice][task]
        expertise = [x / num_of_tasks for x in expertise]
    return [expertise, expertise_score, confidence, confidence_score, difficulty, difficulty_score] #check whether local can return


def check_completed_tasks(num_of_tasks, threshold, infer_difficulty_score, completed_tasks):
    for i in range(num_of_tasks):
        if i not in completed_tasks and infer_difficulty_score[i] >= threshold:
            completed_tasks.append(i)


def synthetic_exp(assign_mode, max_number_of_workers, worker_arri_rate, num_of_tasks, num_of_choices, expertise_init, difficulty_init, confidence_init, threshold):
    num_of_workers = 0
    infer_expertise = []
    infer_expertise_score = []
    infer_confidence = [[confidence_init] * num_of_tasks for _ in range(num_of_choices)]
    infer_confidence_score = [ np.zeros(num_of_tasks) for _ in range(num_of_choices)]
    infer_difficulty = [difficulty_init] * num_of_tasks
    infer_difficulty_score = np.zeros(num_of_tasks)
    truths = tasks_generator(num_of_tasks, num_of_choices) # 1 x tasks
    completed_tasks = []
    time = 0
    while(len(completed_tasks) < num_of_tasks): #begin a batch, old workers/tasks: last batch paras, new workers/tasks: initialized
        # print "len of completed: ", completed_tasks
        check_completed_tasks(num_of_tasks, threshold, infer_difficulty_score, completed_tasks) # check whether completed tasks are updated
        # num_of_workers += worker_arri_rate if num_of_workers < max_number_of_workers else 1 # control the max numer of workers
        num_of_workers += worker_arri_rate
        if num_of_workers == worker_arri_rate:
            assign_scheme_tbw = [np.zeros((num_of_tasks, num_of_workers)) for _ in range(num_of_choices)]  # assignmnet scheme
        else:
            assign_scheme_tbw = [np.hstack((assign_scheme_tbw[i], np.zeros((num_of_tasks, worker_arri_rate)))) for i in range(num_of_choices)]

        infer_expertise = infer_expertise + [expertise_init] * worker_arri_rate
        infer_expertise_score = infer_expertise_score + [-np.log(1-expertise_init)] * worker_arri_rate
        estimated_difficulty_score = np.abs(np.subtract(np.asarray(infer_confidence[0]), np.asarray(infer_confidence[1])))
        task_capacity = threshold - estimated_difficulty_score #threshold can be vector or scala
        [dei_wbt, prob_ans_wbt] = calculate_dei(num_of_workers, num_of_tasks, num_of_choices, infer_expertise, infer_expertise_score,
                                                infer_difficulty, estimated_difficulty_score, infer_confidence, infer_confidence_score)

        # consider task processing time, should add an available worker set, check whether assign_scheme_tbw is changed
        assign_with_mode(assign_mode, num_of_workers, num_of_tasks, task_capacity, dei_wbt, prob_ans_wbt, assign_scheme_tbw, completed_tasks) # assign_scheme_tbw includes the answers

        #start inference return model paras
        [infer_expertise, infer_expertise_score, infer_confidence, infer_confidence_score, infer_difficulty, infer_difficulty_score] = start_inference(num_of_workers, num_of_tasks, num_of_choices, assign_scheme_tbw,expertise_init, difficulty_init)

        time += 1
    print "total time:", time



num_of_tasks = 10
worker_arri_rate = 5
num_of_choices = 2
threshold = 0.8
expertise_init = 0.5
difficulty_init = 0.5
confidence_init = 0.5
max_number_of_workers = 50
synthetic_exp("random", max_number_of_workers, worker_arri_rate, num_of_tasks, num_of_choices, expertise_init, difficulty_init, confidence_init, threshold)

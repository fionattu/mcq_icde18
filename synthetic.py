from truthfinder import *
import numpy as np

# worker selection probability is different from the truthfinder similation
max_number_of_workers = 50


def init(num_of_workers, expertise_init, num_of_tasks, difficulty_init):
    return initialization(num_of_workers, expertise_init, num_of_tasks, difficulty_init)


def prescan(task_capacity, dei_wbt, assign_scheme_tbw): # label completed tasks
    pass


def check_assignment(worker, task, assign_scheme_tbw):
    if assign_scheme_tbw[0][task][worker] is 1 or assign_scheme_tbw[1][task][worker] is 1:
        return True
    else:
        return False


def full_assign(worker, num_of_tasks, assign_scheme_tbw):
    for i in range(num_of_tasks):
        if check_assignment(worker, i, assign_scheme_tbw) is False:
            return False

    return True


def select_task(worker, num_of_tasks, assign_scheme_tbw):
    if full_assign(worker, num_of_tasks, assign_scheme_tbw) is True:
        return -1
    else:
        task = np.random.randint(0, num_of_tasks) #low inclusive, high exclusive
        while(True):
            if check_assignment(worker, task, assign_scheme_tbw) is True:
                task = uniform_random_generator(0, num_of_tasks)
            else:
                return task


def generate_answer(worker, task, prob_ans_wbt): # check normalization is needed?
    if uniform_random_generator(0,1) < prob_ans_wbt[0][worker][task]:
        return 0
    else:
        return 1


def random_assign(num_of_workers, num_of_tasks, prob_ans_wbt, assign_scheme_tbw): #available worker set
    for i in range(num_of_workers):
        task = select_task(i, num_of_tasks, assign_scheme_tbw)
        if task is not -1:
            choice = generate_answer(i, task, prob_ans_wbt) # check whether assignment_scheme_tbw is updated
            assign_scheme_tbw[choice][task][i] = 1
        else:
            pass #this worker has completed all tasks


def assign_with_mode(assign_mode, num_of_workers, num_of_tasks, task_capacity, dei_wbt, prob_ans_wbt, assign_scheme_tbw): # cold start
    if assign_mode is "random":
        random_assign(num_of_workers, num_of_tasks, prob_ans_wbt, assign_scheme_tbw)

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
    prob_ans_wbt = [np.zeros(num_of_workers, num_of_tasks) for _ in range(num_of_choices)]
    prob_ans_wbt[0] = prob_ans_eq_truth_wbt * np.asarray(infer_confidence[0]) + prob_ans_neq_truth_wbt * np.asarray(infer_confidence[1])
    prob_ans_wbt[1] = prob_ans_neq_truth_wbt * np.asarray(infer_confidence[0]) + prob_ans_eq_truth_wbt * np.asarray(infer_confidence[1])

    return prob_ans_wbt


def calculate_ei(infer_confidence_score, infer_confidence, infer_expertise_score, infer_difficulty, estimated_difficulty_score, num_of_workers, num_of_tasks, num_of_choices):
    #infer_expertise_score and infer_difficulty are "1 x tasks"
    ei_wbt = [np.zeros(num_of_workers, num_of_tasks) for _ in range(num_of_choices)]
    updated_confidence_score = [np.zeros(num_of_workers, num_of_tasks) for _ in range(num_of_choices)]
    updated_confidence = [np.zeros(num_of_workers, num_of_tasks) for _ in range(num_of_choices)]
    updated_difficulty_score = [np.zeros(num_of_workers, num_of_tasks) for _ in range(num_of_choices)]
    for i in range(num_of_choices):
        updated_confidence_score[i] = np.add([[infer_confidence_score[i]],] * num_of_workers, np.transpose(infer_expertise_score)) # have to add dampen factors
        updated_confidence[i] = (1/(1 + np.power(math.e, -updated_confidence_score[i]))) * np.asarray(infer_difficulty) #make sure the infer/update confidence have the same difficulties
        if i is 0:
            updated_difficulty_score[i] = np.abs(np.subtract(updated_confidence[i] - infer_confidence[1]))
        elif i is 1:
            updated_difficulty_score[i] = np.abs(np.subtract(updated_confidence[i] - infer_confidence[0]))

        ei_wbt[i] = np.subtract(updated_difficulty_score[i] - estimated_difficulty_score)
    return ei_wbt


def calculate_dei(num_of_workers, num_of_tasks, num_of_choices, infer_expertise, infer_expertise_score, infer_difficulty, estimated_difficulty_score, infer_confidence, infer_confidence_score):
    # infer_confidence_score is 2 x tasks, infer_expertise_score is 1 x workers
    prob_ans_eq_truth_wbt = 1 / (1 + np.power(math.e, -3 * np.dot(np.transpose(infer_expertise), np.transpose(infer_difficulty))))
    prob_ans_wbt = calculate_prob_ans_wbt(prob_ans_eq_truth_wbt, infer_confidence, num_of_workers, num_of_tasks, num_of_choices)
    ei_wbt = calculate_ei(infer_confidence_score, infer_confidence, infer_expertise_score, infer_difficulty, estimated_difficulty_score, num_of_workers, num_of_tasks, num_of_choices)
    dei_wbt = np.abs(np.add(np.multiply(prob_ans_wbt[0],ei_wbt[0]) + np.multiply(prob_ans_wbt[1],ei_wbt[1]))) # should be abs
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

            confidence_score_damping[0][:] = confidence_score[0][:] - 1 * confidence_score[1][:];
            confidence_score_damping[1][:] = confidence_score[1][:] - 1 * confidence_score[0][:];

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


def label_completed_tasks(completed_tasks, infer_confidence):
    pass


def synthetic_exp(assign_mode, max_number_of_workers, worker_arri_rate, num_of_tasks, num_of_choices, expertise_init, difficulty_init, confidence_init, threshold):
    num_of_return_answers = 0
    num_of_workers = 0
    infer_expertise = [expertise_init] * max_number_of_workers
    infer_confidence = [[confidence_init] * num_of_tasks for _ in range(num_of_choices)]
    infer_difficulty = [difficulty_init] * num_of_tasks
    assign_scheme_tbw = [np.zeros(num_of_tasks, max_number_of_workers) for _ in range(num_of_choices)] # assignmnet scheme
    dei_wbt = np.zeros(max_number_of_workers, num_of_tasks)
    task_capacity = np.zeros(num_of_tasks)
    truths = tasks_generator(num_of_tasks, num_of_choices) # 1 x tasks
    completed_tasks = np.zeros(num_of_tasks)
    while(num_of_return_answers < num_of_tasks): #begin a batch
        infer_expertise[0:num_of_workers-1] = expertise
        infer_confidence = confidence
        infer_difficulty = difficulty
        num_of_workers += worker_arri_rate if num_of_workers < max_number_of_workers else 1
        estimated_difficulty_score = np.abs(infer_confidence[0] - infer_confidence[1])
        task_capacity = threshold - estimated_difficulty_score #threshold can be vector or scala
        [dei_wbt, prob_ans_wbt] = calculate_dei(num_of_workers, num_of_tasks, num_of_choices, infer_expertise, infer_expertise_score, infer_difficulty, estimated_difficulty_score, infer_confidence, infer_confidence_score)
        # consider task processing time, should add an available worker set, check whether assign_scheme_tbw is changed
        assign_with_mode(assign_mode, num_of_workers, num_of_tasks, task_capacity, dei_wbt, prob_ans_wbt, assign_scheme_tbw) # assign_scheme_tbw includes the answers
        
        #start assignment
            # calculate_dei(confidence of answer/expertise)

            # assign_with_mode
            #generate and store answers
        #end assignment

        #start inference return model paras
        [infer_expertise, infer_expertise_score, infer_confidence, infer_confidence_score, infer_difficulty,infer_difficulty_score] = start_inference(num_of_workers, num_of_tasks, num_of_choices, assign_scheme_tbw,expertise_init, difficulty_init)
        label_completed_tasks(completed_tasks, infer_confidence) #if certain, stop assign
            # return certain answer (num_of_return_answers = num_of_return_answers + 1)
        #end inference







number_of_tasks = 10
worker_arri_rate = 5
number_of_choices = 2
threshold = 0.8
synthetic_exp(worker_arri_rate, 0.5, number_of_tasks, 0.5, threshold)
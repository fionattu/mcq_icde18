from truthfinder import *
import numpy as np


max_number_of_workers = 50

def init(num_of_workers, expertise_init, num_of_tasks, difficulty_init):
    return initialization(num_of_workers, expertise_init, num_of_tasks, difficulty_init)


def start_inference():
    pass

def assign_with_mode(assignment_mode="random", initial=True):
    if assignment_mode is "random" and initial:
        # cold start
        pass
    elif assignment_mode is "random" and not initial:
        # baseline
        pass
    elif assignment_mode is "firstfit" and not initial:
        pass
    elif assignment_mode is "bestfit" and not initial:
        pass


def calculate_prob_ans_wbt(prob_ans_eq_truth_wbt, infer_confidence, num_of_workers, num_of_tasks, num_of_choices):
    #check element of infer_confidence should be of 1 x tasks
    prob_ans_neq_truth_wbt = (1 - prob_ans_eq_truth_wbt)/(num_of_choices - 1)
    # worker selection probability
    prob_ans_wbt = [np.zeros(num_of_workers, num_of_tasks) for _ in range(num_of_choices)]
    prob_ans_wbt[0] = prob_ans_eq_truth_wbt * infer_confidence[0] + prob_ans_neq_truth_wbt * infer_confidence[1]
    prob_ans_wbt[1] = prob_ans_neq_truth_wbt * infer_confidence[0] + prob_ans_eq_truth_wbt * infer_confidence[1]

    return prob_ans_wbt


def calculate_dei(num_of_workers, num_of_tasks, num_of_choices, infer_expertise, infer_difficulty, infer_confidence):
    prob_ans_eq_truth_wbt = 1 / (1 + np.power(math.e, -3 * np.dot(np.transpose(infer_expertise), np.transpose(infer_difficulty))))
    prob_ans_wbt = calculate_prob_ans_wbt(prob_ans_eq_truth_wbt, infer_confidence, num_of_workers, num_of_tasks, num_of_choices)

    # for i in range(num_of_workers):
    #     for j in range(num_of_tasks):
    #         dei = 0
    #         for k in range(num_of_choices):
    #             dei += (prob_ans * ei)


def synthetic_exp(max_number_of_workers, worker_arri_rate, expertise_init, num_of_tasks, difficulty_init, num_of_choices, confidence_init, threshold):
    num_of_return_answers = 0
    num_of_workers = 0
    infer_expertise = [expertise_init] * max_number_of_workers
    infer_confidence = [[confidence_init] * num_of_tasks for _ in range(num_of_choices)]
    infer_difficulty = [difficulty_init] * num_of_tasks
    assign_scheme_tbw = [np.zeros(num_of_tasks, max_number_of_workers) for _ in range(num_of_choices)] # assignmnet scheme
    dei_wbt = np.zeros(max_number_of_workers, num_of_tasks)
    task_capacity = np.zeros(num_of_tasks)
    while(num_of_return_answers < num_of_tasks): #begin a batch
        infer_expertise[0:num_of_workers-1] = expertise
        infer_confidence = confidence
        infer_difficulty = difficulty
        num_of_workers += worker_arri_rate if num_of_workers < max_number_of_workers else 1

        task_capacity = threshold - np.abs(infer_confidence[0]-infer_confidence[1]) #threshold can be vector or scala

        calculate_dei(num_of_workers, num_of_tasks, num_of_choices, infer_expertise, infer_difficulty, infer_confidence)


        #start assignment
            # calculate_dei(confidence of answer/expertise), generate worker answers
            # assign_with_mode
            #generate and store asnwers
        #end assignment

        #start inference
        [expertise, difficulty] = init(num_of_workers, expertise_init, num_of_tasks, difficulty_init)
            # return certain answer (num_of_return_answers = num_of_return_answers + 1)
        #end inference







number_of_tasks = 10
worker_arri_rate = 5
number_of_choices = 2
threshold = 0.8
synthetic_exp(worker_arri_rate, 0.5, number_of_tasks, 0.5, threshold)
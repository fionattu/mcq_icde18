import numpy as np

from real import *
from truthfinder import uniform_random_generator


def get_available_workers(num_of_workers, batch_workers, assignment):

    ava_workers = range(num_of_workers)
    if len(assignment) == 0:
        return ava_workers
    for ass_index in range(len(assignment)):
        for ass_key in assignment[ass_index]:
            if len(assignment[ass_index][ass_key]) != 0:
                ava_workers.remove(ass_key)
    return ava_workers


def run_qasca(timestamps, workers, truths, ans, num_of_batches, num_of_choices, eval, repetition, k, confidence_init, difficulty_init, expertise_init):
    num_of_workers = 0
    num_of_tasks = len(truths)
    infer_expertise = []
    infer_expertise_score = []
    infer_confidence = [[confidence_init] * num_of_tasks for _ in range(num_of_choices)]
    infer_confidence_score = [np.zeros(num_of_tasks) for _ in range(num_of_choices)]
    infer_difficulty = [difficulty_init] * num_of_tasks
    difficulty_truths = [uniform_random_generator(0.8, 1)] * num_of_tasks
    expertise_truths = []
    infer_truths = np.zeros(num_of_tasks)
    completed_tasks = []
    assign_tbw = np.zeros((num_of_tasks, num_of_workers))
    assign_scheme_tbw = [np.zeros((num_of_tasks, num_of_workers)) for _ in range(num_of_choices)]  # assignmnet scheme

    quality = []
    tasks = []
    processing = []
    repeats = [repetition] * num_of_tasks
    resList = []
    for t in range(num_of_tasks):
        task = [1.0 / num_of_choices] * num_of_choices
        tasks.append(task)

    completed = False
    current_batch = 0
    while (completed is False):
        print "______________________batch______________________", current_batch
        if current_batch <= num_of_batches:
            batch_workers = find_batch_workers(current_batch, timestamps) # add avaliable worker set, worker set can be empty
            if len(batch_workers) != 0:
                print batch_workers
                num_of_new_workers = len(batch_workers) - num_of_workers
                num_of_workers += num_of_workers

                assign_tbw = np.hstack((assign_tbw, np.zeros((num_of_tasks, num_of_new_workers))))
                assign_scheme_tbw = [np.hstack((assign_scheme_tbw[i], np.zeros((num_of_tasks, num_of_new_workers)))) for i in range(num_of_choices)]
                infer_expertise = infer_expertise + [sum(infer_expertise) / (num_of_workers - num_of_new_workers)] * num_of_new_workers

                quality = quality + [uniform_random_generator(0.5, 0.999)] * num_of_new_workers
                available_workers = get_available_workers(num_of_workers, batch_workers, processing)

                # assign(eval, num_of_tasks, available_workers, assign_tbw, quality, tasks, k, processing, repeats)
                # worker_submit_answers(processing, tasks, quality, assign_scheme_tbw, truths,
                #                       resList)  # update prococessing, update qc
                # if check_completed(num_of_tasks, repeats, processing) is True:
                #     completed = True
                # if completed is False:
                #     time += 1
                #     num_of_workers += worker_arri_rate

            else:
                print "no workers arriving"

        else:
            print "number of batches can not complete all tasks"

        current_batch += 1

    # using em as inference
    # em = EM.infer(resList, quality)
    # infer_truths = get_result(num_of_tasks, em)
    # using MCQ inference
    #     [infer_expertise, infer_expertise_score, infer_confidence, infer_confidence_score, infer_difficulty,
    #      infer_difficulty_score] = start_inference(num_of_workers, num_of_tasks, num_of_choices, assign_scheme_tbw,
    #                                                expertise_init, difficulty_init)
    #     infer_truths = get_infer_truths(num_of_tasks, infer_confidence) + 1

    # return [print_accuracy(num_of_tasks, truths, infer_truths), current_batch]
    # qasca_end

def main(ans_dataset='d_Duck Identification_40w217q', arrival_dataset='Relevance_of_terms_to_disaster_relief_topics',matching_mode='random',num_of_batches=40):
    start_and_end_timestamps, workers, truths, ans = batch_assignment(ans_dataset, arrival_dataset, matching_mode, num_of_batches)
    run_qasca(start_and_end_timestamps, workers, truths, ans, num_of_batches, 2, 'accuracy', 3,2,0.5,0.5,0.5)
    print ""
    # baseline(even_timestamps, start_and_end_timestamps, num_of_batches, )
    # ff(even_timestamps, start_and_end_timestamps)
    # bf(even_timestamps, start_and_end_timestamps)
    # random(even_timestamps, start_and_end_timestamps)
    # baseline(even_timestamps, num_of_batches, workers, truths,ans)

num_of_batches = 40
main()
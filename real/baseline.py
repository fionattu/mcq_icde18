import numpy as np

from real.real import find_batch_workers
from truthfinder import uniform_random_generator


def baseline(timestamps, workers, truths, ans, num_of_batches, num_of_choices, eval, repetition, k, confidence_init, difficulty_init, expertise_init):
    num_of_workers = len(workers)
    num_of_tasks = len(truths)
    infer_expertise = [expertise_init] * num_of_workers
    infer_expertise_score = []
    infer_confidence = [[confidence_init] * num_of_tasks for _ in range(num_of_choices)]
    infer_confidence_score = [np.zeros(num_of_tasks) for _ in range(num_of_choices)]
    infer_difficulty = [difficulty_init] * num_of_tasks
    difficulty_truths = [uniform_random_generator(0.8, 1)] * num_of_tasks
    expertise_truths = []
    infer_truths = np.zeros(num_of_tasks)
    completed_tasks = []
    quality = [uniform_random_generator(0.5, 0.999)] * num_of_workers
    tasks = []
    processing = []
    repeats = [repetition] * num_of_tasks
    resList = []
    assign_scheme_tbw = [np.zeros((num_of_tasks, num_of_workers)) for _ in
                         range(num_of_choices)]
    assign_tbw = np.zeros((num_of_tasks, num_of_workers))
    for t in range(num_of_tasks):
        task = [1.0 / num_of_choices] * num_of_choices
        tasks.append(task)

    completed = False
    current_batch = 0
    while (completed is False):
        print "______________________batch______________________", current_batch
        if current_batch <= num_of_batches:
            worker_set = find_batch_workers(current_batch, timestamps) # add avialable worker set
            available_workers = get_available_workers(num_of_workers, processing)
            assign(eval, num_of_tasks, available_workers, assign_tbw, quality, tasks, k, processing, repeats)
            worker_submit_answers(processing, tasks, quality, assign_scheme_tbw, truths,
                                  resList)  # update prococessing, update qc
            if check_completed(num_of_tasks, repeats, processing) is True:
                completed = True
            if completed is False:
                time += 1
                num_of_workers += worker_arri_rate
            current_batch += 1




            # using em as inference


    em = EM.infer(resList, quality)
    infer_truths = get_result(num_of_tasks, em)
    # using MCQ inference
    #     [infer_expertise, infer_expertise_score, infer_confidence, infer_confidence_score, infer_difficulty,
    #      infer_difficulty_score] = start_inference(num_of_workers, num_of_tasks, num_of_choices, assign_scheme_tbw,
    #                                                expertise_init, difficulty_init)
    #     infer_truths = get_infer_truths(num_of_tasks, infer_confidence) + 1

    return [print_accuracy(num_of_tasks, truths, infer_truths), current_batch]
    # qasca_end

